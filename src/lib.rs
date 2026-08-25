//! Bevy host case for Neon3 UI integration.
//!
//! The case deliberately keeps gameplay ECS and Neon UI separate. Bevy owns
//! entities, movement, and camera state. Neon owns NUI declaration, layout,
//! semantic hit resolution, and final UI pixels.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    marker::PhantomData,
    net::SocketAddr,
    sync::{Arc, Mutex, OnceLock, mpsc},
    thread,
    time::{Duration, Instant},
};

use bevy::core_pipeline::{Core3d, Core3dSystems, tonemapping::tonemapping};
use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::{asset::AssetEvent, camera::Projection};

use bevy_render::{
    RenderApp,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::TextureFormat,
    render_resource::{BindGroup, RenderPipeline},
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{ViewDepthTexture, ViewTarget},
};
use neon_ipc::{EventClient, RpcClient};
use neon_protocol::{
    ClientIdentity, ClientKind, EVENT_PROTOCOL, EventFilter, EventFrame, EventSubscribe,
    PROTOCOL_VERSION, RenderSurfaceKind, RenderSurfaceOpen, RenderSurfacePlacement,
    RenderSurfaceSize, RenderSurfaceTarget, RenderSurfaceTargetKind, RequestId, Revision,
    RpcRequest, RpcResponse, ServiceName, UiImageSource, UiImageUploadRequest,
};
use neon_ui::{UiInputChange, UiInputFrame, UiInputValue, UiProgramRevision};
use neon_world_bridge::{
    CameraFrame, CameraFramePayload, CameraId, CoordinateSystem, WorldInformationSnapshot,
    WorldPrecisionMode, WorldSpaceId, WorldUiAnchorBatch, WorldUiAnchorSample,
};
use serde_json::json;

pub mod gpu_readback;

#[cfg(windows)]
mod dx12_consumer;

pub const SCREEN_SURFACE_ID: &str = "case.bevy.screen.ui";
pub const COLOR_TARGET_ID: &str = "case.bevy.screen.ui.color";
pub const WORLD_SURFACE_ID: &str = "case.bevy.world.ui";
pub const WORLD_COLOR_TARGET_ID: &str = "case.bevy.world.ui.color";
pub const UI_BACKING_SCALE: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Neon3ServiceMode {
    /// The host starts and owns the local Neon services.
    #[default]
    AutoHeadless,
    /// The host starts a visible Neon WGPU window and the UI forwarder.
    AutoWindowed,
    /// The application is embedding or supervising Neon services itself.
    External,
}
/// Maximum edge length for an external image source. This is not a thumbnail
/// policy: it matches the renderer-owned atlas width (2048) minus padding, so
/// sources up to this size are uploaded at native resolution without any
/// downscale. Larger sources are only ever met by engine-authored textures;
/// the RPC frame serializes `Vec<u8>` as a JSON array, so `DEFAULT_MAX_FRAME_SIZE`
/// (neon-ipc) must stay large enough for the largest supported image.
const EXTERNAL_UI_IMAGE_MAX_EDGE: u32 = 2046;

/// Camera near/far used for the `camera.submit_frame` projection handed to
/// Neon (Neon rejects world anchors outside [near, far]). The composite
/// UI and scene depth use Bevy's infinite reversed-Z encoding
/// (`near / view_distance`). Keep `near` in sync with the camera's
/// `PerspectiveProjection`.
const SCENE_NEAR: f32 = 0.1;
const SCENE_FAR: f32 = 1000.0;

/// Fullscreen composite shader that samples the Neon color + occlusion depth and
/// discards UI pixels occluded by nearer scene geometry (Bevy's reversed-Z depth
/// prepass).
#[cfg(windows)]
const COMPOSITE_SHADER: &str = r#"
struct Params { near: f32, far: f32, has_scene_depth: u32, debug_mode: u32 }
@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var color_sampler: sampler;
@group(0) @binding(2) var ui_depth_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_depth_sampler: sampler;
@group(0) @binding(4) var scene_depth_tex: texture_depth_2d;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var scene_color_tex: texture_2d<f32>;
@group(0) @binding(7) var scene_color_sampler: sampler;

struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) index: u32) -> Out {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(3.0, 1.0), vec2<f32>(-1.0, 1.0));
    var uvs = array<vec2<f32>, 3>(vec2<f32>(0.0, 2.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 0.0));
    var out: Out;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.uv = uvs[index];
    return out;
}

fn premultiply_sample(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

// The external UI target is straight-alpha RGBA. Filtering it directly lets
// RGB from transparent (usually white) texels bleed into antialiased edges.
// Filter premultiplied values instead, then composite the result ourselves.
fn sample_premultiplied(uv_in: vec2<f32>) -> vec4<f32> {
    let uv = clamp(uv_in, vec2<f32>(0.0), vec2<f32>(1.0));
    let dims_u = textureDimensions(color_tex);
    let dims = vec2<f32>(dims_u);
    let position = uv * dims - vec2<f32>(0.5);
    let base = vec2<i32>(floor(position));
    let fraction = fract(position);
    let max_texel = vec2<i32>(dims_u) - vec2<i32>(1);
    let c00 = premultiply_sample(textureLoad(color_tex, clamp(base, vec2<i32>(0), max_texel), 0));
    let c10 = premultiply_sample(textureLoad(color_tex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), max_texel), 0));
    let c01 = premultiply_sample(textureLoad(color_tex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), max_texel), 0));
    let c11 = premultiply_sample(textureLoad(color_tex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), max_texel), 0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

// Closest scene depth over a 3x3 texel neighborhood, taking the maximum across
// every MSAA sample (reversed-Z: larger is closer). The spatial dilation keeps
// thin occluders (window frames, wires, outlines) from leaking the UI through
// their sub-pixel gaps.
fn scene_depth_msaa_at(texel: vec2<i32>) -> f32 {
    let dims = textureDimensions(scene_depth_tex);
    let last = vec2<i32>(dims) - vec2<i32>(1);
    var depth = 0.0;
    var oy = -1;
    loop {
        var ox = -1;
        loop {
            let sample_texel = clamp(texel + vec2<i32>(ox, oy), vec2<i32>(0), last);
            depth = max(depth, textureLoad(scene_depth_tex, sample_texel, 0));
            ox = ox + 1;
            if (ox > 1) { break; }
        }
        oy = oy + 1;
        if (oy > 1) { break; }
    }
    return depth;
}

@fragment fn fs(input: Out) -> @location(0) vec4<f32> {
    let filtered = sample_premultiplied(input.uv);
    // Screen UI still uses the pipeline's straight-alpha blend state. Convert
    // the filtered premultiplied sample back only for that diagnostic path.
    var color = filtered;
    if (params.debug_mode == 4u && filtered.a > 0.0001) {
        color = vec4<f32>(filtered.rgb / filtered.a, filtered.a);
    }
    let scene_color = textureSample(scene_color_tex, scene_color_sampler, input.uv);
    let ui_dims = vec2<i32>(textureDimensions(ui_depth_tex));
    let ui_texel = clamp(vec2<i32>(input.uv * vec2<f32>(ui_dims)), vec2<i32>(0), ui_dims - vec2<i32>(1));
    let ui_depth = textureLoad(ui_depth_tex, ui_texel, 0).r;
    // A depth-tested World UI edge must have a matching producer depth sample.
    // A zero depth with partial color coverage is a producer/consumer edge
    // mismatch, not an always-visible panel; letting it through creates the
    // white fringe along an occluding building.
    if (params.debug_mode != 4u && ui_depth <= 0.0 && color.a < 0.999) {
        return scene_color;
    }
    // The occlusion depth target only carries projected world UI.
    // 0.0 = always-visible world panel (never occluded by the scene);
    // (0.0, 1.0) = depth-tested world UI in Bevy's reversed-Z encoding,
    // near / view_distance (near -> 1, infinity -> 0). Screen UI never
    // reaches this pass: it is composited on top with no depth participation,
    // so no reserved marker value is needed.
    let dims = vec2<i32>(textureDimensions(scene_depth_tex));
    let texel = clamp(vec2<i32>(input.uv * vec2<f32>(dims)), vec2<i32>(0), dims - vec2<i32>(1));
    let scene_z = scene_depth_msaa_at(texel);
    if (params.debug_mode == 1u) { return vec4<f32>(scene_z, scene_z, scene_z, 1.0); }
    if (params.debug_mode == 2u) { return vec4<f32>(ui_depth, ui_depth, ui_depth, 1.0); }
    if (params.debug_mode == 3u) {
        var result = vec4<f32>(0.0, 1.0, 0.0, 1.0);
        if (ui_depth > 0.0 && params.has_scene_depth == 1u && scene_z > ui_depth) {
            result = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        return result;
    }
    if (color.a <= 0.001) {
        if (params.debug_mode == 4u) { discard; }
        return scene_color;
    }
    if (params.debug_mode == 4u) { return color; }
    // Bevy 0.19 projects the camera with
    // `perspective_infinite_reverse_rh` (bevy_camera::projection,
    // `get_clip_from_view` uses `perspective_infinite_reverse_rh`, so both
    // textures encode near / view_distance. Larger reversed-Z values are
    // closer to the camera; discard only UI hidden by a closer scene pixel.
    let ui_over_scene = vec4<f32>(scene_color.rgb * (1.0 - color.a) + color.rgb, 1.0);
    if (params.has_scene_depth == 0u) { return ui_over_scene; }
    if (ui_depth > 0.0 && params.has_scene_depth == 1u) {
        if (scene_z > ui_depth) { return scene_color; }
    }
    return ui_over_scene;
}
"#;

/// The declarative NUI document for the overhead status component. Bevy submits
/// this source text verbatim; the UI Runtime parses, compiles, activates the
/// host adapter, and forwards the resulting fragment to the renderer.
const ORDINARY_STATUS_NUI: &str = include_str!("../assets/ui/ordinary-status.nui");

nui_flow_vars! {
    CharacterStatusVars => {
        flow: "character-status",
        component: "character.player.main.status",
        fields: {
            health: f32 => "health",
            mana: f32 => "mana",
            level: u32 => "level",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NuiFlowIdentity {
    pub program_revision: UiProgramRevision,
    pub expected_input_revision: Revision,
    pub request_sequence: u64,
}

/// Error returned when applying an authoritative variable writeback to a
/// `NuiFlowVars` value. `UnknownKey` means the flow has no such variable;
/// `TypeMismatch` means the value kind does not match the declared field type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NuiFlowVarError {
    UnknownKey(String),
    TypeMismatch(String),
}

pub trait NuiFlowVars: Clone + PartialEq + Send + Sync + 'static {
    const FLOW_NAME: &'static str;
    const COMPONENT_NAME: &'static str;

    fn snapshot(&self, identity: &mut NuiFlowIdentity) -> UiInputFrame;
    fn diff(&self, previous: &Self, identity: &mut NuiFlowIdentity) -> Option<UiInputFrame>;
    /// Produces one full template row for batched world-UI input. The row is
    /// keyed by the instance anchor (the template's `stable_row_key`) and carries
    /// every declared variable value.
    fn row_snapshot(&self, anchor: &str) -> neon_ui::UiRepeatRow;
    /// Applies an authoritative writeback of variable changes (UI → host). The
    /// macro-generated implementation matches each change key to a field and
    /// type-checks the value before assignment. Unknown keys and wrong kinds
    /// return a stable error instead of silently dropping the change.
    fn apply_changes(&mut self, changes: &[neon_ui::UiInputChange]) -> Result<(), NuiFlowVarError>;
}

#[macro_export]
macro_rules! nui_flow_vars {
    (
        $name:ident => {
            flow: $flow:literal,
            component: $component:literal,
            fields: {
                $( $field:ident : $ty:ty => $key:literal ),+ $(,)?
            }
        }
    ) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            $( pub $field: $ty, )+
        }

        impl $crate::NuiFlowVars for $name {
            const FLOW_NAME: &'static str = $flow;
            const COMPONENT_NAME: &'static str = $component;

            fn snapshot(&self, identity: &mut $crate::NuiFlowIdentity) -> neon_ui::UiInputFrame {
                identity.request_sequence = identity.request_sequence.saturating_add(1);
                neon_ui::UiInputFrame {
                    program_revision: identity.program_revision.clone(),
                    expected_input_revision: identity.expected_input_revision,
                    request_id: format!("nui-flow:{}:{}", $flow, identity.request_sequence),
                    idempotency_key: format!("nui-flow:{}:{}", $flow, identity.request_sequence),
                    changes: vec![
                        $( neon_ui::UiInputChange { key: $key.into(), value: $crate::nui_flow_value!(self.$field) }, )+
                    ],
                }
            }

            fn diff(&self, previous: &Self, identity: &mut $crate::NuiFlowIdentity) -> Option<neon_ui::UiInputFrame> {
                let mut changes = Vec::new();
                $( if self.$field != previous.$field {
                    changes.push(neon_ui::UiInputChange { key: $key.into(), value: $crate::nui_flow_value!(self.$field) });
                } )+
                if changes.is_empty() {
                    return None;
                }
                identity.request_sequence = identity.request_sequence.saturating_add(1);
                Some(neon_ui::UiInputFrame {
                    program_revision: identity.program_revision.clone(),
                    expected_input_revision: identity.expected_input_revision,
                    request_id: format!("nui-flow:{}:{}", $flow, identity.request_sequence),
                    idempotency_key: format!("nui-flow:{}:{}", $flow, identity.request_sequence),
                    changes,
                })
            }

            fn apply_changes(&mut self, changes: &[neon_ui::UiInputChange]) -> Result<(), $crate::NuiFlowVarError> {
                for change in changes {
                    match change.key.as_str() {
                        $( $key => {
                            self.$field = <$ty as $crate::NuiFlowValue>::from_ui_input_value(&change.value)
                                .map_err(|()| $crate::NuiFlowVarError::TypeMismatch(change.key.clone()))?;
                        } )+
                        _ => return Err($crate::NuiFlowVarError::UnknownKey(change.key.clone())),
                    }
                }
                Ok(())
            }

            fn row_snapshot(&self, anchor: &str) -> neon_ui::UiRepeatRow {
                neon_ui::UiRepeatRow {
                    stable_row_key: anchor.into(),
                    values: std::collections::BTreeMap::from([
                        $( ($key.to_string(), $crate::nui_flow_value!(self.$field)), )+
                    ]),
                    semantic_payload: std::collections::BTreeMap::new(),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! nui_flow_value {
    ($value:expr) => {
        $crate::nui_flow_value_ref(&$value)
    };
}

pub fn nui_flow_value_ref<T: NuiFlowValue>(value: &T) -> UiInputValue {
    value.to_ui_input_value()
}

pub trait NuiFlowValue: Sized {
    fn to_ui_input_value(&self) -> UiInputValue;
    fn from_ui_input_value(value: &UiInputValue) -> Result<Self, ()>;
}

impl NuiFlowValue for bool {
    fn to_ui_input_value(&self) -> UiInputValue {
        UiInputValue::Bool { value: *self }
    }
    fn from_ui_input_value(value: &UiInputValue) -> Result<Self, ()> {
        match value {
            UiInputValue::Bool { value } => Ok(*value),
            _ => Err(()),
        }
    }
}
impl NuiFlowValue for i32 {
    fn to_ui_input_value(&self) -> UiInputValue {
        UiInputValue::I32 { value: *self }
    }
    fn from_ui_input_value(value: &UiInputValue) -> Result<Self, ()> {
        match value {
            UiInputValue::I32 { value } => Ok(*value),
            _ => Err(()),
        }
    }
}
impl NuiFlowValue for u32 {
    fn to_ui_input_value(&self) -> UiInputValue {
        UiInputValue::U32 { value: *self }
    }
    fn from_ui_input_value(value: &UiInputValue) -> Result<Self, ()> {
        match value {
            UiInputValue::U32 { value } => Ok(*value),
            _ => Err(()),
        }
    }
}
impl NuiFlowValue for f32 {
    fn to_ui_input_value(&self) -> UiInputValue {
        UiInputValue::F32 { value: *self }
    }
    fn from_ui_input_value(value: &UiInputValue) -> Result<Self, ()> {
        match value {
            UiInputValue::F32 { value } => Ok(*value),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Neon3BevyConfig {
    pub wgpu_endpoint: SocketAddr,
    pub ui_endpoint: SocketAddr,
    pub eventd_endpoint: Option<SocketAddr>,
    pub session_id: String,
    pub surface_size: [u32; 2],
    pub ui_sources: Vec<String>,
    pub world_ui: bool,
    pub service_mode: Neon3ServiceMode,
    pub auto_start_services: bool,
}

impl Default for Neon3BevyConfig {
    fn default() -> Self {
        Self {
            wgpu_endpoint: "127.0.0.1:39103"
                .parse()
                .expect("valid default WGPU endpoint"),
            ui_endpoint: "127.0.0.1:39102"
                .parse()
                .expect("valid default UI endpoint"),
            eventd_endpoint: None,
            session_id: "bevy-nui-host".into(),
            surface_size: [1280 * UI_BACKING_SCALE, 720 * UI_BACKING_SCALE],
            ui_sources: vec![ORDINARY_STATUS_NUI.into()],
            world_ui: false,
            service_mode: Neon3ServiceMode::AutoHeadless,
            auto_start_services: true,
        }
    }
}

#[derive(Resource, Debug)]
pub struct Neon3Session {
    pub config: Neon3BevyConfig,
    pub surface_id: String,
    pub color_target_id: String,
    pub generation: u64,
    pub producer_epoch: u64,
    pub frame_sequence: u64,
    /// One sequence is shared by the camera frame and its anchor batch. The
    /// latest-value resource assigns it only when a complete world snapshot
    /// changes, so the two payloads can never drift into independent streams.
    pub world_frame_sequence: u64,
    pub connected: bool,
    pub last_error: Option<String>,
    pub acquire_requested: bool,
    pub surface_acquired: bool,
    pub world_surface_acquired: bool,
    pub world_ui_snapshot: Option<Vec<neon_ui::UiRepeatRow>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorldFrameSignature {
    camera_position: [u32; 3],
    camera_orientation: [u32; 4],
    fov: u32,
    near: u32,
    far: u32,
    anchor_count: u32,
    anchor_hash: u64,
}

#[derive(Clone, Debug)]
struct PendingCameraSubmission {
    frame: CameraFrame,
    signature: WorldFrameSignature,
    dirty: bool,
    in_flight: bool,
}

#[derive(Clone, Debug)]
struct PendingAnchorSubmission {
    batch: WorldUiAnchorBatch,
    signature: WorldFrameSignature,
    dirty: bool,
    in_flight: bool,
}

/// Latest-value state for the world data plane. Pending values are overwritten
/// while a request is in flight; only the newest dirty value is sent next.
#[derive(Resource, Default)]
struct LatestWorldSubmission {
    camera: Option<PendingCameraSubmission>,
    anchors: Option<PendingAnchorSubmission>,
    last_sent_signature: Option<WorldFrameSignature>,
    last_sent_camera: Option<CameraFrame>,
    last_sent_anchors: Option<WorldUiAnchorBatch>,
}

#[derive(Resource, Default)]
pub struct Neon3IntentQueue {
    pub events: Vec<Neon3Intent>,
}

/// Typed latest-value input changes supplied by a host case. The UI runtime
/// remains the owner of revision validation and fragment presentation.
#[derive(Resource, Default)]
pub struct Neon3InputChanges {
    pub changes: BTreeMap<String, UiInputValue>,
}

#[derive(Clone, Debug)]
pub struct Neon3VariableEvent {
    pub name: String,
    pub epoch: u64,
    pub sequence: u64,
    pub payload: serde_json::Value,
}

#[derive(Resource, Default)]
pub struct Neon3VariableEvents {
    pub events: Vec<Neon3VariableEvent>,
}

#[derive(Resource, Debug)]
pub struct Neon3ExternalSurfaceGpu {
    pub surface_id: String,
    pub color_target_id: String,
    pub size: [u32; 2],
    pub generation: u64,
    pub frame_sequence: u64,
    pub color_format: TextureFormat,
    pub imported: bool,
    #[cfg(windows)]
    pub imported_colors: Vec<Neon3ImportedColorBuffer>,
    #[cfg(windows)]
    pub imported_world_colors: Vec<Neon3ImportedColorBuffer>,
    #[cfg(windows)]
    pub imported_depths: Vec<Neon3ImportedDepthBuffer>,
    #[cfg(windows)]
    pub imported_world_depths: Vec<Neon3ImportedDepthBuffer>,
    #[cfg(windows)]
    pub local_depth: Option<Neon3LocalDepthTarget>,
    #[cfg(windows)]
    pub composite: Option<Neon3CompositePipeline>,
}

#[cfg(windows)]
#[derive(Debug)]
pub struct Neon3ImportedColorBuffer {
    pub buffer_index: u32,
    pub texture: dx12_consumer::ImportedTexture,
}

#[cfg(windows)]
#[derive(Debug)]
pub struct Neon3ImportedDepthBuffer {
    pub buffer_index: u32,
    pub texture: dx12_consumer::ImportedTexture,
}

#[cfg(windows)]
#[derive(Debug)]
pub struct Neon3LocalDepthTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

#[cfg(windows)]
#[derive(Debug)]
pub struct Neon3CompositePipeline {
    pub pipeline: RenderPipeline,
    pub screen_pipeline: RenderPipeline,
    pub layout: wgpu::BindGroupLayout,
    pub params_buffer: wgpu::Buffer,
    pub screen_params_buffer: wgpu::Buffer,
    pub color_sampler: wgpu::Sampler,
    pub depth_sampler: wgpu::Sampler,
    pub dummy_scene_depth: wgpu::TextureView,
    pub dummy_scene_color: wgpu::TextureView,
}

#[derive(Clone, Debug)]
pub struct Neon3ExternalSurfaceBufferHandle {
    pub buffer_index: u32,
    pub color_texture_handle: usize,
    pub color_fence_handle: usize,
    pub consumer_release_fence_handle: usize,
    pub depth_texture_handle: Option<usize>,
    pub depth_fence_handle: Option<usize>,
    pub depth_consumer_release_fence_handle: Option<usize>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct Neon3ExternalSurfaceHandles {
    pub buffers: Vec<Neon3ExternalSurfaceBufferHandle>,
    pub color_texture_handle: Option<usize>,
    pub color_fence_handle: Option<usize>,
}

/// Bevy-owned image asset exposed to a Neon Flow image node.
///
/// The Handle remains entirely inside Bevy. Only decoded RGBA8 bytes cross the
/// protocol boundary, where the WGPU runtime owns atlas residency.
#[derive(Component)]
pub struct Neon3ExternalImage {
    pub image_id: String,
    pub handle: Handle<Image>,
    /// Optional source crop in source pixels: x, y, width, height.
    pub source_region: Option<[u32; 4]>,
    uploaded_fingerprint: Option<u64>,
    pending_fingerprint: Option<u64>,
    upload_in_flight: bool,
}

impl Neon3ExternalImage {
    pub fn new(image_id: impl Into<String>, handle: Handle<Image>) -> Self {
        Self {
            image_id: image_id.into(),
            handle,
            source_region: None,
            uploaded_fingerprint: None,
            pending_fingerprint: None,
            upload_in_flight: false,
        }
    }

    pub fn with_region(mut self, region: [u32; 4]) -> Self {
        self.source_region = Some(region);
        self
    }
}

#[derive(Resource, Default)]
struct Neon3FlowSubmissionState {
    submitted: HashSet<usize>,
}

#[derive(Resource, Default)]
struct Neon3ImageUploadState {
    next_sequence: u64,
    pending: HashMap<String, Entity>,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct Neon3WorldExternalSurfaceHandles {
    pub buffers: Vec<Neon3ExternalSurfaceBufferHandle>,
}

#[derive(Resource, Clone)]
struct Neon3EventQueue(Arc<Mutex<Vec<Neon3VariableEvent>>>);

#[derive(Resource)]
pub struct CharacterStatusBridge {
    pub identity: NuiFlowIdentity,
    pub current: CharacterStatusVars,
    pub sent: Option<CharacterStatusVars>,
    pub flow_ready: bool,
    pub input_pending: bool,
}

#[derive(Clone, Debug)]
pub struct Neon3Intent {
    pub action: String,
    pub params: serde_json::Value,
    pub request_id: String,
}

/// A zero-sized ECS tag identifying the semantic interaction contract of a UI.
/// The label belongs to the Rust type, not to per-entity runtime data.
pub trait Neon3SemanticInteractionKey: Component + Send + Sync + 'static {
    const KEY: &'static str;
}

#[macro_export]
macro_rules! neon_semantic_key {
    ($name:ident, $key:literal) => {
        #[derive(Component, Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $crate::Neon3SemanticInteractionKey for $name {
            const KEY: &'static str = $key;
        }
    };
}

neon_semantic_key!(CharacterStatusScreenKey, "character.player.main.status");
neon_semantic_key!(MonsterStatusWorldKey, "monster.status");

/// Fixed-screen UI state. Unlike `NeonWorldUi`, it has no world anchor; attach
/// a zero-sized `Neon3SemanticInteractionKey` tag to the same entity so ECS
/// systems can query the UI by a stable semantic type.
#[derive(Component)]
pub struct NeonScreenUi<V: NuiFlowVars> {
    pub flow: String,
    pub vars: V,
    pub identity: NuiFlowIdentity,
    pub sent: Option<V>,
    pub visible: bool,
    pub selected_object: Option<String>,
    pub _marker: PhantomData<fn() -> V>,
}

impl<V: NuiFlowVars> NeonScreenUi<V> {
    pub fn new(flow: impl Into<String>, vars: V, identity: NuiFlowIdentity) -> Self {
        Self {
            flow: flow.into(),
            vars,
            identity,
            sent: None,
            visible: true,
            selected_object: None,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Neon3SemanticIntentEvent {
    pub intent: String,
    pub source_node_key: Option<String>,
    pub interaction_key: Option<String>,
    pub payload: serde_json::Value,
    pub requested_value: Option<serde_json::Value>,
    pub request_id: String,
}

#[derive(Resource, Default)]
pub struct Neon3SemanticIntentEvents {
    pub events: Vec<Neon3SemanticIntentEvent>,
}

impl Neon3SemanticIntentEvents {
    pub fn drain(&mut self) -> std::vec::Drain<'_, Neon3SemanticIntentEvent> {
        self.events.drain(..)
    }
}

pub fn semantic_intent_targets<K: Neon3SemanticInteractionKey>(
    event: &Neon3SemanticIntentEvent,
) -> bool {
    event.interaction_key.as_deref() == Some(K::KEY)
}

#[derive(Resource, Default)]
pub struct Neon3PointerEvents {
    pub events: Vec<Neon3PointerEvent>,
}

#[derive(Clone, Debug)]
pub struct Neon3PointerEvent {
    pub event_type: String,
    pub pixel: Option<[u32; 2]>,
    pub delta: [f32; 2],
    pub delta_mode: String,
    pub button: Option<String>,
    pub buttons: Vec<String>,
    pub modifiers: Vec<String>,
    pub pointer_id: u64,
    pub sequence: u64,
    pub generation: u64,
    pub frame_sequence: u64,
}

#[derive(Resource, Default)]
struct Neon3PointerState {
    next_sequence: u64,
    inside: bool,
    focused: bool,
    last_pixel: Option<[u32; 2]>,
}

#[derive(Component, Clone, Debug)]
pub struct Neon3HostObject {
    pub object_id: String,
}

#[derive(Component)]
pub struct Neon3WalkableCharacter;

/// How a world-UI panel participates in scene occlusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WorldUiOcclusion {
    /// Never depth-tested: the panel is always drawn on top of scene geometry.
    #[default]
    AlwaysVisible,
    /// Depth-tested against scene depth: pixels occluded by nearer geometry are
    /// discarded during Bevy's composite. Requires the UI depth target path.
    DepthTested,
}

impl WorldUiOcclusion {
    /// Protocol string used on `WorldUiAnchor.occlusion`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AlwaysVisible => "always_visible",
            Self::DepthTested => "depth_tested",
        }
    }
}

/// A world-space UI instance. The component is generic over the bound variable
/// struct `V` (which implements `NuiFlowVars`): its `vars` field *is* the set of
/// flow variables, and many entities share one flow template while each owns its
/// own `vars` + `anchor`. Each instance maps to one template row keyed by
/// `anchor` in a batched `UiRepeatFrame`.
#[derive(Component)]
pub struct NeonWorldUi<V: NuiFlowVars> {
    /// Flow source name (identifies the shared template / program).
    pub flow: String,
    /// Live variable values — the component's "fields" are the variable fields.
    pub vars: V,
    /// Per-instance identity (program/input revision, request sequence).
    pub identity: NuiFlowIdentity,
    /// Stable instance key (= template `stable_row_key`); usually the anchor id.
    pub anchor: String,
    /// World-space offset from the entity transform (billboard anchor point).
    pub offset: Vec3,
    /// Last submitted snapshot, used for sparse diff.
    pub sent: Option<V>,
    /// Frustum-culling / visibility state (observational, not authoritative).
    pub visible: bool,
    /// Scene-occlusion policy for this panel (see `WorldUiOcclusion`).
    pub occlusion: WorldUiOcclusion,
}

impl<V: NuiFlowVars> NeonWorldUi<V> {
    /// The stable per-instance key used in batched `UiRepeatFrame` rows.
    pub fn stable_row_key(&self) -> String {
        self.anchor.clone()
    }

    /// Returns the full, typed template row for this visible instance. The
    /// caller batches rows from all entities into one UiRepeatFrame.
    pub fn row_snapshot(&self) -> Option<neon_ui::UiRepeatRow> {
        self.visible.then(|| self.vars.row_snapshot(&self.anchor))
    }
}

/// Collects the independent ECS-owned world UI instances without changing
/// their identity or deriving any domain values. A boundary system may batch
/// the returned rows for transport after this step.
pub fn collect_world_ui_rows<'a, V, I>(world_uis: I) -> Vec<neon_ui::UiRepeatRow>
where
    V: NuiFlowVars + 'a,
    I: IntoIterator<Item = &'a NeonWorldUi<V>>,
{
    world_uis
        .into_iter()
        .filter_map(NeonWorldUi::row_snapshot)
        .collect()
}

#[derive(Resource)]
struct Neon3Transport {
    requests: mpsc::Sender<TransportRequest>,
    responses: Arc<Mutex<mpsc::Receiver<TransportResponse>>>,
}

#[derive(Resource)]
struct Neon3OwnedServices {
    threads: Vec<std::thread::JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RpcLane {
    WorldLatest,
    Interaction,
}

struct TransportRequest {
    endpoint: SocketAddr,
    request: RpcRequest,
    lane: RpcLane,
}

struct TransportResponse {
    lane: RpcLane,
    request_id: RequestId,
    result: Result<RpcResponse, String>,
}

pub struct Neon3BevyPlugin {
    config: Neon3BevyConfig,
}

impl Neon3BevyPlugin {
    pub fn new(config: Neon3BevyConfig) -> Self {
        Self { config }
    }
}

impl Default for Neon3BevyPlugin {
    fn default() -> Self {
        Self::new(Neon3BevyConfig::default())
    }
}

impl Plugin for Neon3BevyPlugin {
    fn build(&self, app: &mut App) {
        let owned_services = if self.config.auto_start_services
            && self.config.service_mode != Neon3ServiceMode::External
        {
            Some(start_owned_services(&self.config))
        } else {
            None
        };
        let (request_tx, request_rx) = mpsc::channel::<TransportRequest>();
        let (response_tx, response_rx) = mpsc::channel::<TransportResponse>();
        let endpoint = self.config.wgpu_endpoint;
        thread::Builder::new()
            .name("neon3-bevy-rpc".into())
            .spawn(move || transport_worker(endpoint, request_rx, response_tx))
            .expect("start Neon3 Bevy transport worker");

        let variable_events = Arc::new(Mutex::new(Vec::new()));
        if let Some(eventd_endpoint) = self.config.eventd_endpoint {
            let events_for_thread = Arc::clone(&variable_events);
            thread::Builder::new()
                .name("neon3-bevy-eventd".into())
                .spawn(move || eventd_worker(eventd_endpoint, events_for_thread))
                .expect("start Neon3 Bevy eventd worker");
        }

        app.insert_resource(Neon3Session {
            config: self.config.clone(),
            surface_id: SCREEN_SURFACE_ID.into(),
            color_target_id: COLOR_TARGET_ID.into(),
            generation: 0,
            producer_epoch: 1,
            frame_sequence: 0,
            world_frame_sequence: 0,
            connected: false,
            last_error: None,
            acquire_requested: false,
            surface_acquired: false,
            world_surface_acquired: false,
            world_ui_snapshot: None,
        })
        .insert_resource(Neon3OwnedServices {
            threads: owned_services.unwrap_or_default(),
        })
        .insert_resource(LatestWorldSubmission::default())
        .insert_resource(Neon3IntentQueue::default())
        .insert_resource(Neon3InputChanges::default())
        .insert_resource(Neon3PointerEvents::default())
        .insert_resource(Neon3PointerState::default())
        .insert_resource(Neon3SemanticIntentEvents::default())
        .insert_resource(Neon3VariableEvents::default())
        .insert_resource(Neon3EventQueue(variable_events))
        .insert_resource(Neon3ImageUploadState::default())
        .insert_resource(Neon3FlowSubmissionState::default())
        .insert_resource(CharacterStatusBridge {
            identity: NuiFlowIdentity {
                program_revision: UiProgramRevision {
                    program_id: "character.player.main.status".into(),
                    revision: Revision(1),
                    schema_version: neon_ui::UI_PROGRAM_SCHEMA_VERSION,
                    capabilities: Vec::new(),
                },
                expected_input_revision: Revision(0),
                request_sequence: 0,
            },
            current: CharacterStatusVars {
                health: 82.0,
                mana: 64.0,
                level: 12,
            },
            sent: None,
            flow_ready: false,
            input_pending: false,
        })
        .insert_resource(Neon3Transport {
            requests: request_tx,
            responses: Arc::new(Mutex::new(response_rx)),
        })
        .add_systems(
            Startup,
            (
                request_world_info,
                request_screen_surface,
                request_world_surface,
            ),
        )
        .add_systems(
            Update,
            (
                consume_neon_responses,
                consume_variable_events,
                upload_external_images,
                submit_flow_source,
                publish_pointer_events,
                publish_camera_snapshot,
                publish_world_anchor,
                flush_world_latest,
                flush_intents,
                flush_pointer_events,
                flush_input_changes,
                flush_character_status,
            )
                .chain(),
        );
        app.insert_resource(Neon3ExternalSurfaceHandles {
            buffers: Vec::new(),
            color_texture_handle: None,
            color_fence_handle: None,
        });
        app.insert_resource(Neon3WorldExternalSurfaceHandles {
            buffers: Vec::new(),
        });
        app.add_plugins(ExtractResourcePlugin::<Neon3ExternalSurfaceHandles>::default());
        app.add_plugins(ExtractResourcePlugin::<Neon3WorldExternalSurfaceHandles>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(Neon3ExternalSurfaceGpu {
                surface_id: SCREEN_SURFACE_ID.into(),
                color_target_id: COLOR_TARGET_ID.into(),
                size: self.config.surface_size,
                generation: 0,
                frame_sequence: 0,
                color_format: TextureFormat::Rgba8Unorm,
                imported: false,
                #[cfg(windows)]
                imported_colors: Vec::new(),
                imported_world_colors: Vec::new(),
                #[cfg(windows)]
                imported_depths: Vec::new(),
                imported_world_depths: Vec::new(),
                local_depth: None,
                #[cfg(windows)]
                composite: None,
            });
            // Seed the render world with the handles resource so the render
            // system validates on the first frame; the ExtractResourcePlugin
            // keeps it in sync with the main world on every subsequent frame.
            render_app.insert_resource(Neon3ExternalSurfaceHandles {
                buffers: Vec::new(),
                color_texture_handle: None,
                color_fence_handle: None,
            });
            render_app.insert_resource(Neon3WorldExternalSurfaceHandles {
                buffers: Vec::new(),
            });
            render_app.add_systems(
                Core3d,
                neon3_external_surface_render_system
                    .after(tonemapping)
                    .in_set(Core3dSystems::PostProcess),
            );
        }
    }
}

fn start_owned_services(config: &Neon3BevyConfig) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();
    let eventd = config.eventd_endpoint.map(|endpoint| {
        std::thread::Builder::new()
            .name("neon3-bevy-owned-eventd".into())
            .spawn(move || {
                if let Err(error) = neon_eventd::serve(endpoint, 1) {
                    eprintln!("[neon3-bevy-plugin] eventd stopped: {error}");
                }
            })
            .expect("start owned Neon eventd")
    });
    if let Some(thread) = eventd {
        threads.push(thread);
    }
    let wgpu_endpoint = config.wgpu_endpoint;
    match config.service_mode {
        Neon3ServiceMode::AutoHeadless => {
            #[cfg(windows)]
            threads.push(
                std::thread::Builder::new()
                    .name("neon3-bevy-owned-headless-wgpu".into())
                    .spawn(move || {
                        let handle =
                            neon_wgpu_runtime::spawn_headless_external_server(wgpu_endpoint);
                        let _ = handle.join();
                    })
                    .expect("start owned headless Neon WGPU"),
            );
            #[cfg(not(windows))]
            panic!("AutoHeadless Neon3 WGPU service currently requires Windows");
        }
        Neon3ServiceMode::AutoWindowed => {
            let ui_endpoint = config.ui_endpoint;
            threads.push(
                std::thread::Builder::new()
                    .name("neon3-bevy-owned-windowed-wgpu".into())
                    .spawn(move || {
                        if let Err(error) = neon_wgpu_runtime::WindowedRuntime::run_server(
                            1,
                            wgpu_endpoint,
                            Some(ui_endpoint),
                            None,
                            false,
                        ) {
                            eprintln!("[neon3-bevy-plugin] windowed WGPU stopped: {error}");
                        }
                    })
                    .expect("start owned windowed Neon WGPU"),
            );
        }
        Neon3ServiceMode::External => {}
    }
    wait_for_owned_service(config.wgpu_endpoint, "wgpu-runtime");
    let ui_endpoint = config.ui_endpoint;
    let wgpu_endpoint = config.wgpu_endpoint;
    let domain_endpoint = config.ui_endpoint;
    let eventd_endpoint = config.eventd_endpoint;
    threads.push(
        std::thread::Builder::new()
            .name("neon3-bevy-owned-ui".into())
            .spawn(move || {
                if let Err(error) = neon_ui::serve_forwarder(
                    ui_endpoint,
                    wgpu_endpoint,
                    domain_endpoint,
                    eventd_endpoint,
                    1,
                ) {
                    eprintln!("[neon3-bevy-plugin] UI forwarder stopped: {error}");
                }
            })
            .expect("start owned Neon UI forwarder"),
    );
    wait_for_owned_service(config.ui_endpoint, "ui-runtime");
    threads
}

fn wait_for_owned_service(endpoint: SocketAddr, target: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let request = RpcRequest {
            protocol: neon_protocol::RPC_PROTOCOL.into(),
            version: PROTOCOL_VERSION,
            request_id: RequestId(format!("bevy-plugin-health-{target}")),
            client: ClientIdentity {
                kind: ClientKind::ExternalHost,
                instance_id: "neon3-bevy-plugin-startup".into(),
                pid: std::process::id(),
                origin: "neon3-bevy-plugin".into(),
            },
            target: ServiceName(target.into()),
            method: "service.health".into(),
            params: json!({}),
            expected_revision: None,
            idempotency_key: Some(format!("bevy-plugin-health:{target}")),
        };
        if let Ok(mut client) = RpcClient::connect(endpoint)
            .and_then(|client| client.with_timeout(Duration::from_millis(250)))
            && client.call(&request).is_ok()
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("[neon3-bevy-plugin] {target} did not become healthy at {endpoint}");
}

fn neon3_external_surface_render_system(
    mut surface: ResMut<Neon3ExternalSurfaceGpu>,
    handles: Res<Neon3ExternalSurfaceHandles>,
    world_handles: Res<Neon3WorldExternalSurfaceHandles>,
    _render_device: Res<RenderDevice>,
    _render_queue: Res<RenderQueue>,
    views: ViewQuery<(&ViewTarget, &ViewDepthTexture)>,
    mut render_context: RenderContext,
    mut pending_releases: Local<Vec<(u32, u64)>>,
    mut overlay_disabled_reported: Local<bool>,
    mut last_selected_external_frame: Local<Option<(u32, u64)>>,
    mut screen_bind_groups: Local<HashMap<u32, BindGroup>>,
    // Per-frame post-process resources that only need rebuilding when the
    // scene-depth state or the viewport size changes. The bind group and the
    // params buffer contents are otherwise stable for the lifetime of a frame
    // size, so rebuilding/re-writing them every frame is pure overhead.
    mut last_scene_depth_state: Local<Option<u32>>,
) {
    // Copy the selected completed external frame into a Bevy-owned texture
    // before the post-process overlay. The main scene never samples the foreign
    // D3D12 resource directly.
    if std::env::var("NEON3_DISABLE_EXTERNAL_UI").as_deref() == Ok("1") {
        if !*overlay_disabled_reported {
            warn!("Neon UI overlay disabled by NEON3_DISABLE_EXTERNAL_UI=1");
            *overlay_disabled_reported = true;
        }
        return;
    }
    // Screen UI is independent of world-surface startup. Do not suppress the
    // final overlay while the world ring is still being imported or has not
    // produced its first completed frame.
    if handles.buffers.is_empty() {
        return;
    }
    let (view_target, view_depth) = views.into_inner();
    let output_format = view_target.main_texture_format();
    let post_process = view_target.post_process_write();
    let scene_color_view = post_process.source;
    #[cfg(windows)]
    if !surface.imported {
        if let Some(buffer) = handles.buffers.first() {
            match dx12_consumer::import_texture(
                _render_device.wgpu_device(),
                buffer.color_texture_handle,
                buffer.color_fence_handle,
                buffer.consumer_release_fence_handle,
                surface.size,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(imported) => {
                    let device = _render_device.wgpu_device();
                    let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
                    let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("neon3-bevy-ui-depth-sampler"),
                        mag_filter: wgpu::FilterMode::Nearest,
                        min_filter: wgpu::FilterMode::Nearest,
                        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                        ..default()
                    });
                    let layout =
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("neon3-bevy-external-ui-layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Float {
                                            filterable: true,
                                        },
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Filtering,
                                    ),
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Float {
                                            filterable: false,
                                        },
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::NonFiltering,
                                    ),
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Depth,
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 6,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Texture {
                                        sample_type: wgpu::TextureSampleType::Float {
                                            filterable: true,
                                        },
                                        view_dimension: wgpu::TextureViewDimension::D2,
                                        multisampled: false,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 7,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Sampler(
                                        wgpu::SamplerBindingType::Filtering,
                                    ),
                                    count: None,
                                },
                            ],
                        });
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("neon3-bevy-external-ui-shader"),
                        source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
                    });
                    let pipeline_layout =
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("neon3-bevy-external-ui-pipeline-layout"),
                            bind_group_layouts: &[Some(&layout)],
                            immediate_size: 0,
                        });
                    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("neon3-bevy-external-ui-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: output_format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: Default::default(),
                        depth_stencil: None,
                        multisample: Default::default(),
                        multiview_mask: None,
                        cache: None,
                    });
                    let screen_pipeline =
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("neon3-bevy-screen-ui-overlay-pipeline"),
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs"),
                                buffers: &[],
                                compilation_options: Default::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: output_format,
                                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: Default::default(),
                            }),
                            primitive: Default::default(),
                            depth_stencil: None,
                            multisample: Default::default(),
                            multiview_mask: None,
                            cache: None,
                        });
                    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("neon3-bevy-ui-params"),
                        size: 16,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let screen_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("neon3-bevy-screen-ui-params"),
                        size: 16,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let local_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("neon3-bevy-ui-depth-fallback"),
                        size: wgpu::Extent3d {
                            width: surface.size[0],
                            height: surface.size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let local_depth_view =
                        local_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let dummy_scene_depth = device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("neon3-bevy-dummy-scene-depth"),
                            size: wgpu::Extent3d {
                                width: 1,
                                height: 1,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Depth32Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        })
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let dummy_scene_color = device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("neon3-bevy-dummy-scene-color"),
                            size: wgpu::Extent3d {
                                width: 1,
                                height: 1,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        })
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    surface.composite = Some(Neon3CompositePipeline {
                        pipeline: RenderPipeline::from(pipeline),
                        screen_pipeline: RenderPipeline::from(screen_pipeline),
                        layout,
                        params_buffer,
                        screen_params_buffer,
                        color_sampler,
                        depth_sampler,
                        dummy_scene_depth,
                        dummy_scene_color,
                    });
                    surface.imported_colors.push(Neon3ImportedColorBuffer {
                        buffer_index: buffer.buffer_index,
                        texture: imported,
                    });
                    surface.local_depth = Some(Neon3LocalDepthTarget {
                        texture: local_depth_texture,
                        view: local_depth_view,
                    });
                    surface.imported = true;
                }
                Err(error) => {
                    warn!("Neon: failed to import external color texture: {error}");
                }
            }
        }
    }
    #[cfg(windows)]
    if surface.imported && surface.imported_colors.len() < handles.buffers.len() {
        let device = _render_device.wgpu_device();
        for buffer in &handles.buffers {
            if surface
                .imported_colors
                .iter()
                .any(|imported| imported.buffer_index == buffer.buffer_index)
            {
                continue;
            }
            let Ok(imported) = dx12_consumer::import_texture(
                device,
                buffer.color_texture_handle,
                buffer.color_fence_handle,
                buffer.consumer_release_fence_handle,
                surface.size,
                wgpu::TextureFormat::Rgba8Unorm,
            ) else {
                continue;
            };
            surface.imported_colors.push(Neon3ImportedColorBuffer {
                buffer_index: buffer.buffer_index,
                texture: imported,
            });
        }
    }
    #[cfg(windows)]
    if surface.imported && surface.imported_world_colors.len() < world_handles.buffers.len() {
        let device = _render_device.wgpu_device();
        for buffer in &world_handles.buffers {
            if surface
                .imported_world_colors
                .iter()
                .any(|imported| imported.buffer_index == buffer.buffer_index)
            {
                continue;
            }
            let Ok(imported) = dx12_consumer::import_texture(
                device,
                buffer.color_texture_handle,
                buffer.color_fence_handle,
                buffer.consumer_release_fence_handle,
                surface.size,
                wgpu::TextureFormat::Rgba8Unorm,
            ) else {
                continue;
            };
            surface
                .imported_world_colors
                .push(Neon3ImportedColorBuffer {
                    buffer_index: buffer.buffer_index,
                    texture: imported,
                });
        }
    }
    #[cfg(windows)]
    if surface.imported && surface.imported_depths.len() < handles.buffers.len() {
        let device = _render_device.wgpu_device();
        for buffer in &handles.buffers {
            if surface
                .imported_depths
                .iter()
                .any(|imported| imported.buffer_index == buffer.buffer_index)
            {
                continue;
            }
            let Some(depth_handle) = buffer.depth_texture_handle else {
                continue;
            };
            // The depth ring carries its own per-buffer fence and consumer-release
            // fence. The producer waits on the *depth* consumer-release fence before
            // reusing a slot (its free-buffer check requires BOTH the color and depth
            // release fences to advance), so importing it with the color fence here
            // would leave the depth release fence permanently un-signaled and starve
            // the producer into dropping every frame once the ring fills.
            let (Some(depth_fence_handle), Some(depth_release_handle)) = (
                buffer.depth_fence_handle,
                buffer.depth_consumer_release_fence_handle,
            ) else {
                continue;
            };
            let Ok(imported) = dx12_consumer::import_texture(
                device,
                depth_handle,
                depth_fence_handle,
                depth_release_handle,
                surface.size,
                wgpu::TextureFormat::R32Float,
            ) else {
                continue;
            };
            surface.imported_depths.push(Neon3ImportedDepthBuffer {
                buffer_index: buffer.buffer_index,
                texture: imported,
            });
        }
    }
    #[cfg(windows)]
    if surface.imported && surface.imported_world_depths.len() < world_handles.buffers.len() {
        let device = _render_device.wgpu_device();
        for buffer in &world_handles.buffers {
            if surface
                .imported_world_depths
                .iter()
                .any(|imported| imported.buffer_index == buffer.buffer_index)
            {
                continue;
            }
            let (Some(depth_handle), Some(depth_fence_handle), Some(depth_release_handle)) = (
                buffer.depth_texture_handle,
                buffer.depth_fence_handle,
                buffer.depth_consumer_release_fence_handle,
            ) else {
                continue;
            };
            let Ok(imported) = dx12_consumer::import_texture(
                device,
                depth_handle,
                depth_fence_handle,
                depth_release_handle,
                surface.size,
                wgpu::TextureFormat::R32Float,
            ) else {
                continue;
            };
            surface
                .imported_world_depths
                .push(Neon3ImportedDepthBuffer {
                    buffer_index: buffer.buffer_index,
                    texture: imported,
                });
        }
    }
    #[cfg(windows)]
    let completed_frames = surface
        .imported_colors
        .iter()
        .map(|buffer| {
            (
                buffer.buffer_index,
                dx12_consumer::completed_fence_value(&buffer.texture),
            )
        })
        .collect::<Vec<_>>();
    #[cfg(windows)]
    #[cfg(windows)]
    let selected = completed_frames
        .iter()
        .enumerate()
        .filter_map(|(index, (_buffer_index, screen_frame))| {
            let buffer_index = surface.imported_colors[index].buffer_index;
            let world_color_ready = surface.imported_world_colors.is_empty()
                || surface
                    .imported_world_colors
                    .iter()
                    .find(|color| color.buffer_index == buffer_index)
                    .is_some_and(|color| {
                        dx12_consumer::completed_fence_value(&color.texture) >= *screen_frame
                    });
            let world_depth_ready = surface.imported_world_depths.is_empty()
                || surface
                    .imported_world_depths
                    .iter()
                    .find(|depth| depth.buffer_index == buffer_index)
                    .is_some_and(|depth| {
                        dx12_consumer::completed_fence_value(&depth.texture) >= *screen_frame
                    });
            (*screen_frame != 0 && world_color_ready && world_depth_ready)
                .then_some((index, *screen_frame))
        })
        .max_by_key(|(_, screen_frame)| *screen_frame);
    #[cfg(windows)]
    if let Some((selected_index, frame_sequence)) = selected {
        let buffer_index = surface.imported_colors[selected_index].buffer_index;
        let selected_frame = (buffer_index, frame_sequence);
        if std::env::var("NEON3_TRACE_CONSUMER").as_deref() == Ok("1")
            && *last_selected_external_frame != Some(selected_frame)
        {
            eprintln!(
                "{}",
                json!({
                    "event": "bevy_external_ui_consumer_selected",
                    "surface_id": surface.surface_id,
                    "buffer_index": buffer_index,
                    "frame_sequence": frame_sequence,
                    "previous": *last_selected_external_frame,
                })
            );
            *last_selected_external_frame = Some(selected_frame);
        }
    }
    #[cfg(windows)]
    if let Some((selected_index, _)) = selected {
        let selected_buffer_index = surface.imported_colors[selected_index].buffer_index;
        let mut keep = Vec::new();
        for (buffer_index, value) in pending_releases.drain(..) {
            if buffer_index == selected_buffer_index {
                // Keep the selected external texture sampled by this frame.
                keep.push((buffer_index, value));
            } else {
                signal_imported_buffer_release(&surface, &_render_queue, buffer_index, value);
            }
        }
        *pending_releases = keep;
    }
    #[cfg(windows)]
    if let Some((selected_index, completed)) = selected {
        if completed == 0 {
            return;
        }
        let imported = &surface.imported_colors[selected_index];
        if let Err(error) =
            dx12_consumer::wait_external_fence(&_render_queue, &imported.texture, completed)
        {
            warn!("Neon: failed to enqueue shared texture fence wait: {error}");
            return;
        }
        if let Some(depth) = surface
            .imported_depths
            .iter()
            .find(|depth| depth.buffer_index == imported.buffer_index)
            && let Err(error) =
                dx12_consumer::wait_external_fence(&_render_queue, &depth.texture, completed)
        {
            warn!("Neon: failed to enqueue shared depth fence wait: {error}");
            return;
        }
        // The selection above requires world color/depth to have completed the
        // screen frame sequence. Wait for that same value instead of their
        // latest independent completion, which could otherwise mix UI color
        // and occlusion depth from different producer frames.
        if let Some(world) = surface
            .imported_world_colors
            .iter()
            .find(|world| world.buffer_index == imported.buffer_index)
        {
            if let Err(error) =
                dx12_consumer::wait_external_fence(&_render_queue, &world.texture, completed)
            {
                warn!("Neon: failed to enqueue shared world UI fence wait: {error}");
                return;
            }
        }
        if let Some(world_depth) = surface
            .imported_world_depths
            .iter()
            .find(|depth| depth.buffer_index == imported.buffer_index)
        {
            if let Err(error) =
                dx12_consumer::wait_external_fence(&_render_queue, &world_depth.texture, completed)
            {
                warn!("Neon: failed to enqueue shared world depth fence wait: {error}");
                return;
            }
        }
        surface.frame_sequence = completed;
    }
    let Some(composite) = surface.composite.as_ref() else {
        return;
    };
    let mut world_bind_group = None;
    let mut screen_bind_group = None;
    {
        #[cfg(windows)]
        let Some((selected_index, _)) = selected else {
            return;
        };
        let selected_world_color = surface.imported_world_colors.iter().find(|color| {
            color.buffer_index == surface.imported_colors[selected_index].buffer_index
        });
        if let Some(selected_color) = selected_world_color {
            let Some(local_depth) = surface.local_depth.as_ref() else {
                return;
            };
            // Bind the imported ring textures directly. The producer fence wait
            // above orders writes before sampling; the deferred release below
            // keeps the slot alive until this queue has consumed the frame.
            let selected_depth = surface
                .imported_world_depths
                .iter()
                .find(|depth| depth.buffer_index == selected_color.buffer_index);
            // Bevy 0.19 renders the depth prepass into the same texture as the
            // main opaque pass (`view_depth_texture`), so the main depth
            // texture is the authoritative scene depth at post-process time.
            // Its clear value is `Camera3d::depth_load_op` (explicitly
            // `Clear(0.0)` = far plane in reversed-Z, set in the host camera).
            let scene_depth_view = Some(view_depth.view());
            let debug_mode = std::env::var("NEON3_UI_OCCLUSION_DEBUG")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|mode| *mode <= 4)
                .unwrap_or(0);
            let has_scene_depth = if std::env::var("NEON3_DISABLE_UI_OCCLUSION").as_deref()
                != Ok("1")
                && selected_depth.is_some()
                && scene_depth_view.is_some()
            {
                1u32
            } else {
                0u32
            };
            // Re-write when either scene-depth availability or diagnostic mode
            // changes. The fourth u32 is otherwise ignored by normal rendering.
            let state_key = has_scene_depth | (debug_mode << 1);
            if *last_scene_depth_state != Some(state_key) {
                let mut params = [0u8; 16];
                params[0..4].copy_from_slice(&SCENE_NEAR.to_le_bytes());
                params[4..8].copy_from_slice(&SCENE_FAR.to_le_bytes());
                params[8..12].copy_from_slice(&has_scene_depth.to_le_bytes());
                params[12..16].copy_from_slice(&debug_mode.to_le_bytes());
                _render_queue.write_buffer(&composite.params_buffer, 0, &params);
                *last_scene_depth_state = Some(state_key);
            }
            let scene_depth_binding = if has_scene_depth == 1 {
                scene_depth_view.map_or(&composite.dummy_scene_depth, |view| &**view)
            } else {
                &composite.dummy_scene_depth
            };
            let device = _render_device.wgpu_device();
            // The bind group is stable while the scene-depth flag and viewport
            // size are unchanged. Rebuild it only then (e.g. on resize, which
            // recreates the main depth texture and thus `scene_depth_view`).
            let bind_group = {
                let built =
                    BindGroup::from(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("neon3-bevy-external-ui-bind-group"),
                        layout: &composite.layout,
                        entries:
                            &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &selected_color.texture.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        &composite.color_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource:
                                        wgpu::BindingResource::TextureView(
                                            selected_depth.map_or(&local_depth.view, |depth| {
                                                &depth.texture.view
                                            }),
                                        ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(
                                        &composite.depth_sampler,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        scene_depth_binding,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: composite.params_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::Sampler(
                                        &composite.color_sampler,
                                    ),
                                },
                            ],
                    }));
                built
            };
            world_bind_group = Some(bind_group);
        }
    }
    // Screen UI is normally the second layer of the published frame. Keep it
    // disabled by default while diagnosing world UI occlusion; opt back in
    // with NEON3_SHOW_SCREEN_UI=1.
    #[cfg(windows)]
    if let Some((selected_index, _)) = selected {
        let Some(composite) = surface.composite.as_ref() else {
            return;
        };
        let screen_color = &surface.imported_colors[selected_index];
        let mut params = [0u8; 16];
        params[0..4].copy_from_slice(&SCENE_NEAR.to_le_bytes());
        params[4..8].copy_from_slice(&SCENE_FAR.to_le_bytes());
        params[12..16].copy_from_slice(&4u32.to_le_bytes());
        _render_queue.write_buffer(&composite.screen_params_buffer, 0, &params);
        let bind_group =
            if let Some(bind_group) = screen_bind_groups.get(&screen_color.buffer_index) {
                bind_group.clone()
            } else {
                let device = _render_device.wgpu_device();
                let bind_group =
                    BindGroup::from(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("neon3-bevy-screen-ui-bind-group"),
                        layout: &composite.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &screen_color.texture.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&composite.color_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &composite.dummy_scene_depth,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&composite.depth_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(
                                    &composite.dummy_scene_depth,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: composite.screen_params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(
                                    &composite.dummy_scene_color,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::Sampler(&composite.color_sampler),
                            },
                        ],
                    }));
                screen_bind_groups.insert(screen_color.buffer_index, bind_group.clone());
                bind_group
            };
        screen_bind_group = Some(bind_group);
    }
    let mut pass = render_context.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("neon3-bevy-external-ui-overlay"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(&composite.pipeline);
    if let Some(bind_group) = world_bind_group.as_ref() {
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    if let Some(bind_group) = screen_bind_group.as_ref() {
        pass.set_render_pipeline(&composite.screen_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    #[cfg(windows)]
    {
        // Queue all completed slots for the next frame. That frame filters out
        // the slot it is still sampling and releases every other slot, so the
        // producer can reuse old ring entries without racing the current pass.
        *pending_releases = completed_frames
            .iter()
            .filter(|(_, completed)| *completed != 0)
            .copied()
            .collect();
    }
}

#[cfg(windows)]
fn signal_imported_buffer_release(
    surface: &Neon3ExternalSurfaceGpu,
    render_queue: &RenderQueue,
    buffer_index: u32,
    value: u64,
) {
    if let Some(buffer) = surface
        .imported_colors
        .iter()
        .find(|buffer| buffer.buffer_index == buffer_index)
        && let Err(error) =
            dx12_consumer::signal_consumer_release(render_queue, &buffer.texture, value)
    {
        warn!("Neon: failed to signal shared color consumer release: {error}");
    }
    if let Some(depth) = surface
        .imported_depths
        .iter()
        .find(|buffer| buffer.buffer_index == buffer_index)
        && let Err(error) =
            dx12_consumer::signal_consumer_release(render_queue, &depth.texture, value)
    {
        warn!("Neon: failed to signal shared depth consumer release: {error}");
    }
    if let Some(world) = surface
        .imported_world_colors
        .iter()
        .find(|buffer| buffer.buffer_index == buffer_index)
        && let Err(error) =
            dx12_consumer::signal_consumer_release(render_queue, &world.texture, value)
    {
        warn!("Neon: failed to signal shared world color consumer release: {error}");
    }
    if let Some(depth) = surface
        .imported_world_depths
        .iter()
        .find(|buffer| buffer.buffer_index == buffer_index)
        && let Err(error) =
            dx12_consumer::signal_consumer_release(render_queue, &depth.texture, value)
    {
        warn!("Neon: failed to signal shared world depth consumer release: {error}");
    }
}

fn eventd_worker(endpoint: SocketAddr, events: Arc<Mutex<Vec<Neon3VariableEvent>>>) {
    let Ok(mut client) = EventClient::connect(endpoint) else {
        return;
    };
    let subscribe = EventSubscribe {
        protocol: EVENT_PROTOCOL.into(),
        version: PROTOCOL_VERSION,
        request_id: RequestId("bevy-variable-subscribe".into()),
        client: ClientIdentity {
            kind: ClientKind::ExternalHost,
            instance_id: "bevy-nui-host".into(),
            pid: std::process::id(),
            origin: "neon3-bevy-nui-host".into(),
        },
        filters: vec![EventFilter {
            name: None,
            name_prefix: Some("flow.".into()),
            publisher_kinds: None,
        }],
        replay_from_sequence: None,
        max_rate_hz: None,
    };
    let value = serde_json::to_value(EventFrame::Subscribe(subscribe)).unwrap_or_default();
    if client.send_value(&value).is_err() {
        return;
    }
    while let Ok(value) = client.recv_value() {
        let Some(event) = value.get("event") else {
            continue;
        };
        let Some(name) = event.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let item = Neon3VariableEvent {
            name: name.into(),
            epoch: event
                .get("epoch")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default(),
            sequence: event
                .get("sequence")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default(),
            payload: event.get("payload").cloned().unwrap_or_default(),
        };
        if let Ok(mut queue) = events.lock() {
            const MAX_PENDING_VARIABLE_EVENTS: usize = 4096;
            if queue.len() >= MAX_PENDING_VARIABLE_EVENTS {
                let remove_count = queue.len() - MAX_PENDING_VARIABLE_EVENTS + 1;
                queue.drain(..remove_count);
            }
            queue.push(item);
        }
    }
}

fn consume_variable_events(mut output: ResMut<Neon3VariableEvents>, queue: Res<Neon3EventQueue>) {
    let Ok(mut pending) = queue.0.lock() else {
        return;
    };
    output.events.extend(pending.drain(..));
    const MAX_VARIABLE_EVENTS: usize = 4096;
    if output.events.len() > MAX_VARIABLE_EVENTS {
        let remove_count = output.events.len() - MAX_VARIABLE_EVENTS;
        output.events.drain(..remove_count);
    }
}

fn transport_worker(
    _endpoint: SocketAddr,
    requests: mpsc::Receiver<TransportRequest>,
    responses: mpsc::Sender<TransportResponse>,
) {
    // A route is one independent RpcClient connection. In particular, the
    // world-latest and interaction routes to the same WGPU endpoint never
    // share a FIFO or a socket.
    let mut routes = HashMap::<(SocketAddr, RpcLane), mpsc::Sender<TransportRequest>>::new();
    for transport_request in requests {
        let key = (transport_request.endpoint, transport_request.lane);
        let route = if let Some(route) = routes.get(&key) {
            route.clone()
        } else {
            let (request_tx, request_rx) = mpsc::channel();
            let response_tx = responses.clone();
            let (endpoint, lane) = key;
            thread::Builder::new()
                .name(format!("neon3-bevy-rpc-{endpoint}-{lane:?}"))
                .spawn(move || rpc_endpoint_worker(endpoint, lane, request_rx, response_tx))
                .expect("start Neon3 endpoint worker");
            routes.insert(key, request_tx.clone());
            request_tx
        };
        if route.send(transport_request).is_err() {
            routes.remove(&key);
        }
    }
}

fn rpc_endpoint_worker(
    endpoint: SocketAddr,
    lane: RpcLane,
    requests: mpsc::Receiver<TransportRequest>,
    responses: mpsc::Sender<TransportResponse>,
) {
    for transport_request in requests {
        let request_id = transport_request.request.request_id.clone();
        // neon-ipc RpcServer intentionally serves one request per TCP
        // connection. Keep the lane worker independent, but do not reuse a
        // socket after its one response; otherwise the next world-surface or
        // UI request can stall on a normally closed server connection.
        let mut client = None;
        let result = call_rpc_with_reconnect(endpoint, &mut client, &transport_request.request);
        if responses
            .send(TransportResponse {
                lane,
                request_id,
                result,
            })
            .is_err()
        {
            break;
        }
    }
}

fn call_rpc_with_reconnect(
    endpoint: SocketAddr,
    client: &mut Option<RpcClient>,
    request: &RpcRequest,
) -> Result<RpcResponse, String> {
    if client.is_none() {
        *client = Some(
            RpcClient::connect(endpoint)
                .and_then(|client| client.with_timeout(std::time::Duration::from_secs(5)))
                .map_err(|error| error.to_string())?,
        );
    }

    match client
        .as_mut()
        .expect("RPC client initialized")
        .call(request)
    {
        Ok(response) => Ok(response),
        Err(first_error) => {
            // A failed call invalidates the stream. Requests carry idempotency
            // keys, so retry once after reconnecting instead of losing a frame.
            *client = None;
            *client = Some(
                RpcClient::connect(endpoint)
                    .and_then(|client| client.with_timeout(std::time::Duration::from_secs(5)))
                    .map_err(|reconnect_error| {
                        format!("{}; reconnect failed: {reconnect_error}", first_error)
                    })?,
            );
            client
                .as_mut()
                .expect("RPC client reinitialized")
                .call(request)
                .map_err(|retry_error| format!("{first_error}; retry failed: {retry_error}"))
        }
    }
}

fn request_screen_surface(mut session: ResMut<Neon3Session>, transport: Res<Neon3Transport>) {
    let request = rpc_request_for(
        "bevy-surface-open",
        "render.surface.open",
        &session,
        session.config.wgpu_endpoint,
        json!(RenderSurfaceOpen {
            session_id: session.config.session_id.clone(),
            surface_id: session.surface_id.clone(),
            kind: RenderSurfaceKind::ScreenUi,
            size: RenderSurfaceSize {
                width: session.config.surface_size[0],
                height: session.config.surface_size[1],
            },
            format: "rgba8unorm".into(),
            color_space: neon_protocol::RenderSurfaceColorSpace::Linear,
            // Screen UI is the topmost, depth-free layer: it is composited
            // after the scene and after the depth-occluded world UI, so the
            // producer never needs a depth ring for the screen surface.
            depth: false,
            // The in-process headless external server requires a 2- or
            // 3-buffer ring. The Bevy consumer imports and releases these
            // shared buffers asynchronously.
            buffer_count: 3,
            placement: None,
            targets: vec![RenderSurfaceTarget {
                target_id: session.color_target_id.clone(),
                kind: RenderSurfaceTargetKind::Color,
                format: "rgba8unorm".into(),
            },],
        }),
    );
    if transport
        .requests
        .send(TransportRequest {
            endpoint: session.config.wgpu_endpoint,
            request,
            lane: RpcLane::WorldLatest,
        })
        .is_err()
    {
        session.last_error = Some("neon_transport_closed".into());
    }
}

fn request_world_surface(session: Res<Neon3Session>, transport: Res<Neon3Transport>) {
    let request = rpc_request_for(
        "bevy-world-surface-open",
        "render.surface.open",
        &session,
        session.config.wgpu_endpoint,
        json!(RenderSurfaceOpen {
            session_id: session.config.session_id.clone(),
            surface_id: WORLD_SURFACE_ID.into(),
            kind: RenderSurfaceKind::WorldUi,
            size: RenderSurfaceSize {
                width: session.config.surface_size[0],
                height: session.config.surface_size[1]
            },
            format: "rgba8unorm".into(),
            color_space: neon_protocol::RenderSurfaceColorSpace::Linear,
            depth: std::env::var("NEON3_DISABLE_UI_OCCLUSION").as_deref() != Ok("1"),
            buffer_count: 3,
            placement: Some(RenderSurfacePlacement {
                anchor_id: None,
                position: None,
                rotation: None,
                scale: None,
                billboard: true,
                occlusion: "depth_tested".into(),
            }),
            targets: vec![RenderSurfaceTarget {
                target_id: WORLD_COLOR_TARGET_ID.into(),
                kind: RenderSurfaceTargetKind::Color,
                format: "rgba8unorm".into()
            }],
        }),
    );
    let _ = transport.requests.send(TransportRequest {
        endpoint: session.config.wgpu_endpoint,
        request,
        lane: RpcLane::WorldLatest,
    });
}

fn request_world_info(session: Res<Neon3Session>, transport: Res<Neon3Transport>) {
    let request = rpc_request_for(
        "bevy-world-info",
        "wgpu.world.info.configure",
        &session,
        session.config.wgpu_endpoint,
        json!(WorldInformationSnapshot {
            world_space_id: WorldSpaceId("case.bevy.world.main".into()),
            revision: Revision(1),
            coordinate_system: CoordinateSystem::RightHandedYUpNegativeZForward,
            units_per_meter: 1.0,
            precision_mode: WorldPrecisionMode::CameraRelativeF64,
        }),
    );
    let _ = transport.requests.send(TransportRequest {
        endpoint: session.config.wgpu_endpoint,
        request,
        lane: RpcLane::WorldLatest,
    });
}

/// Converts a CPU-resident Bevy image into the Neon external-image wire value.
///
/// Bevy's GPU texture is deliberately not exposed here. This boundary accepts
/// only one-level 2D RGBA8 data that the Neon WGPU owner can place in its atlas.
pub fn neon3_image_source(image_id: impl Into<String>, image: &Image) -> Option<UiImageSource> {
    neon3_image_source_with_region(image_id, image, None)
}

pub fn neon3_image_source_with_region(
    image_id: impl Into<String>,
    image: &Image,
    source_region: Option<[u32; 4]>,
) -> Option<UiImageSource> {
    let size = image.texture_descriptor.size;
    if size.depth_or_array_layers != 1 || image.texture_descriptor.mip_level_count != 1 {
        return None;
    }
    if !matches!(
        image.texture_descriptor.format,
        bevy::render::render_resource::TextureFormat::Rgba8Unorm
            | bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb
    ) {
        return None;
    }
    let data = image.data.as_deref()?;
    let expected = (size.width as usize)
        .checked_mul(size.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))?;
    if data.len() != expected || size.width == 0 || size.height == 0 {
        return None;
    }
    let (source_x, source_y, source_width, source_height) =
        if let Some([x, y, width, height]) = source_region {
            let end_x = x.checked_add(width)?;
            let end_y = y.checked_add(height)?;
            if width == 0 || height == 0 || end_x > size.width || end_y > size.height {
                return None;
            }
            (x, y, width, height)
        } else {
            (0, 0, size.width, size.height)
        };
    let largest_dimension = source_width.max(source_height);
    let (width, height) = if largest_dimension > EXTERNAL_UI_IMAGE_MAX_EDGE {
        (
            ((u64::from(source_width) * u64::from(EXTERNAL_UI_IMAGE_MAX_EDGE))
                / u64::from(largest_dimension))
            .max(1) as u32,
            ((u64::from(source_height) * u64::from(EXTERNAL_UI_IMAGE_MAX_EDGE))
                / u64::from(largest_dimension))
            .max(1) as u32,
        )
    } else {
        (source_width, source_height)
    };
    let bytes = if source_region.is_none() && (width, height) == (size.width, size.height) {
        data.to_vec()
    } else {
        let output_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))?;
        let mut output = Vec::with_capacity(output_len);
        for y in 0..height {
            let source_y = source_y as usize
                + (u64::from(y) * u64::from(source_height) / u64::from(height)) as usize;
            for x in 0..width {
                let source_x = source_x as usize
                    + (u64::from(x) * u64::from(source_width) / u64::from(width)) as usize;
                let source_offset = source_y
                    .checked_mul(size.width as usize)
                    .and_then(|row| row.checked_add(source_x))
                    .and_then(|pixel| pixel.checked_mul(4))?;
                output.extend_from_slice(&data[source_offset..source_offset + 4]);
            }
        }
        output
    };
    Some(UiImageSource {
        image_id: image_id.into(),
        media_type: "application/x-neon-rgba8".into(),
        width,
        height,
        bytes,
    })
}

fn upload_external_images(
    mut images: Query<(Entity, &mut Neon3ExternalImage)>,
    assets: Res<Assets<Image>>,
    mut image_events: MessageReader<AssetEvent<Image>>,
    session: Res<Neon3Session>,
    transport: Res<Neon3Transport>,
    mut state: ResMut<Neon3ImageUploadState>,
) {
    let changed_assets = image_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => Some(id),
            AssetEvent::Removed { .. } | AssetEvent::Unused { .. } => None,
        })
        .collect::<HashSet<_>>();
    for (entity, mut external) in &mut images {
        if changed_assets.contains(&external.handle.id()) {
            external.uploaded_fingerprint = None;
        }
        if external.upload_in_flight {
            continue;
        }
        // A stable Bevy asset does not need a CPU copy, hash, alpha scan, or
        // renderer upload every frame. Asset events above reopen this path when
        // the handle's pixels actually change.
        if external.uploaded_fingerprint.is_some() {
            continue;
        }
        let Some(image) = assets.get(&external.handle) else {
            continue;
        };
        let Some(source) = neon3_image_source_with_region(
            external.image_id.clone(),
            image,
            external.source_region,
        ) else {
            continue;
        };
        let fingerprint = source
            .bytes
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            });
        let alpha_values = source.bytes.chunks_exact(4).map(|pixel| pixel[3]);
        let alpha_nonzero = alpha_values.clone().filter(|alpha| *alpha > 0).count();
        let alpha_max = alpha_values.max().unwrap_or(0);
        if external.uploaded_fingerprint == Some(fingerprint) {
            continue;
        }
        state.next_sequence = state.next_sequence.saturating_add(1);
        let request_id = format!("bevy-image-upload-{}", state.next_sequence);
        eprintln!(
            "{}",
            json!({
                "event": "bevy_external_image_upload_queued",
                "request_id": request_id,
                "image_id": external.image_id,
                "width": source.width,
                "height": source.height,
                "bytes": source.bytes.len(),
                "alpha_nonzero_pixels": alpha_nonzero,
                "alpha_max": alpha_max,
            })
        );
        let request = rpc_request_for(
            request_id.clone(),
            "ui.image.upload",
            &session,
            session.config.ui_endpoint,
            json!(UiImageUploadRequest { source }),
        );
        if transport
            .requests
            .send(TransportRequest {
                endpoint: session.config.ui_endpoint,
                request,
                lane: RpcLane::Interaction,
            })
            .is_ok()
        {
            external.pending_fingerprint = Some(fingerprint);
            external.upload_in_flight = true;
            state.pending.insert(request_id, entity);
        }
    }
}

fn submit_flow_source(
    session: Res<Neon3Session>,
    transport: Res<Neon3Transport>,
    images: Query<&Neon3ExternalImage>,
    mut state: ResMut<Neon3FlowSubmissionState>,
) {
    let images_ready = images
        .iter()
        .all(|image| image.uploaded_fingerprint.is_some());
    for (index, source) in session.config.ui_sources.iter().enumerate() {
        if state.submitted.contains(&index) {
            continue;
        }
        if source.contains("resource ") && !images_ready {
            continue;
        }
        let request = rpc_request_for(
            format!("bevy-flow-source-{}", index),
            "ui.flow.submit",
            &session,
            session.config.ui_endpoint,
            json!({ "source": source }),
        );
        if transport
            .requests
            .send(TransportRequest {
                endpoint: session.config.ui_endpoint,
                request,
                lane: RpcLane::Interaction,
            })
            .is_err()
        {
            warn!("Neon UI flow source request queue is closed");
        } else {
            state.submitted.insert(index);
        }
    }
}

fn semantic_intent_from_result(
    result: &serde_json::Value,
    request_id: &str,
) -> Option<Neon3SemanticIntentEvent> {
    // WGPU pointer resolution returns `semantic_event`; UI Runtime host
    // publications use `semantic_intent`. Both are the same semantic
    // contract at this boundary and must reach the ECS intent system.
    let event = result
        .get("semantic_intent")
        .or_else(|| result.get("semantic_event"))
        .or_else(|| {
            result
                .get("renderer")
                .and_then(|value| value.get("semantic_event"))
        })
        .or_else(|| {
            result
                .get("result")
                .and_then(|value| value.get("semantic_intent"))
        })
        .unwrap_or(result);
    let intent = event
        .get("intent")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            event
                .get("intent")
                .and_then(|intent| intent.get("action"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })?;
    let source_node_key = event
        .get("source_node_key")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let mut payload = event.get("payload").cloned().unwrap_or_else(|| json!({}));
    let requested_value = event.get("requested_value").cloned();
    let interaction_key = event
        .get("interaction_key")
        .or_else(|| event.get("semantic_key"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            let source = source_node_key.as_deref()?;
            if source == "status-root" || source == "status-action" {
                Some("character.player.main.status".into())
            } else if let Some(index) = source.strip_prefix('p').filter(|value| !value.is_empty()) {
                if index.chars().all(|character| character.is_ascii_digit()) {
                    if let Some(object) = payload.as_object_mut() {
                        let anchor = format!("monster.m{index}");
                        object
                            .entry("anchor")
                            .or_insert_with(|| json!(anchor.clone()));
                        object.entry("object_id").or_insert_with(|| json!(anchor));
                    }
                    Some("monster.status".into())
                } else {
                    None
                }
            } else {
                None
            }
        });
    Some(Neon3SemanticIntentEvent {
        intent,
        source_node_key,
        interaction_key,
        payload,
        requested_value,
        request_id: request_id.into(),
    })
}

fn consume_neon_responses(
    mut session: ResMut<Neon3Session>,
    mut latest: ResMut<LatestWorldSubmission>,
    transport: Res<Neon3Transport>,
    mut image_uploads: ResMut<Neon3ImageUploadState>,
    mut handles: ResMut<Neon3ExternalSurfaceHandles>,
    mut world_handles: ResMut<Neon3WorldExternalSurfaceHandles>,
    mut bridge: ResMut<CharacterStatusBridge>,
    mut semantic_events: ResMut<Neon3SemanticIntentEvents>,
    mut external_images: Query<&mut Neon3ExternalImage>,
    mut screens: Query<&mut NeonScreenUi<CharacterStatusVars>, With<CharacterStatusScreenKey>>,
) {
    let Ok(responses) = transport.responses.lock() else {
        session.last_error = Some("neon_response_lock_poisoned".into());
        return;
    };
    while let Ok(transport_response) = responses.try_recv() {
        let TransportResponse {
            lane,
            request_id,
            result: response_result,
        } = transport_response;
        let image_entity = image_uploads.pending.remove(&request_id.0);
        match response_result {
            Ok(response) if response.status == neon_protocol::RpcStatus::Accepted => {
                if let Some(entity) = image_entity
                    && let Ok(mut image) = external_images.get_mut(entity)
                {
                    image.uploaded_fingerprint = image.pending_fingerprint.take();
                    image.upload_in_flight = false;
                    eprintln!(
                        "{}",
                        json!({
                            "event": "bevy_external_image_resident",
                            "request_id": request_id.0,
                            "image_id": image.image_id,
                            "texture": response.result.as_ref().and_then(|result| result.get("texture")),
                        })
                    );
                }
                if request_id.0.starts_with("bevy-camera-") {
                    if let Some(camera) = latest.camera.as_mut() {
                        camera.in_flight = false;
                    }
                }
                if request_id.0.starts_with("bevy-anchor-batch-") {
                    if let Some(anchors) = latest.anchors.as_mut() {
                        anchors.in_flight = false;
                    }
                }
                session.connected = true;
                session.last_error = None;
                if request_id.0.starts_with("bevy-pointer-") {
                    eprintln!(
                        "{}",
                        json!({
                            "event": "bevy_pointer_response",
                            "request_id": request_id.0,
                            "lane": format!("{lane:?}"),
                            "accepted": true,
                            "has_semantic_event": response
                                .result
                                .as_ref()
                                .is_some_and(|result| {
                                    result.get("semantic_event").is_some()
                                        || result.get("semantic_intent").is_some()
                                }),
                            "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                        })
                    );
                }
                if request_id.0.starts_with("nui-flow:") {
                    bridge.input_pending = false;
                }
                if let Some(result) = response.result {
                    if request_id.0.starts_with("bevy-pointer-") {
                        // The pointer already resolved through the UI runtime:
                        // the accepted response carries the resolved semantic
                        // intent directly. Extract it so ECS systems can react
                        // (e.g. apply a physics impulse to the hit object).
                        if let Some(event) = semantic_intent_from_result(&result, &request_id.0) {
                            if event.intent.starts_with("phys.") {
                                eprintln!(
                                    "{}",
                                    json!({
                                        "event": "physics_semantic_enqueued",
                                        "intent": event.intent,
                                        "source_node_key": event.source_node_key,
                                        "requested_value": event.requested_value,
                                        "request_id": event.request_id,
                                        "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                                    })
                                );
                            }
                            const MAX_SEMANTIC_EVENTS: usize = 4096;
                            if semantic_events.events.len() >= MAX_SEMANTIC_EVENTS {
                                semantic_events.events.remove(0);
                            }
                            semantic_events.events.push(event);
                        } else if result.get("semantic_intent").is_some()
                            || result.get("semantic_event").is_some()
                        {
                            eprintln!(
                                "{}",
                                json!({
                                    "event": "physics_semantic_response_unparsed",
                                    "request_id": request_id,
                                    "result": result,
                                    "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                                })
                            );
                        }
                    } else if !request_id.0.starts_with("bevy-pointer-ui-") {
                        if let Some(event) = semantic_intent_from_result(&result, &request_id.0) {
                            const MAX_SEMANTIC_EVENTS: usize = 4096;
                            if semantic_events.events.len() >= MAX_SEMANTIC_EVENTS {
                                semantic_events.events.remove(0);
                            }
                            semantic_events.events.push(event);
                        }
                    }
                    session.generation = result
                        .get("generation")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(session.generation);
                    session.producer_epoch = result
                        .get("producer_epoch")
                        .and_then(serde_json::Value::as_u64)
                        .filter(|epoch| *epoch > 0)
                        .unwrap_or(session.producer_epoch);
                    if request_id.0 == "bevy-surface-open" && !session.acquire_requested {
                        let request = rpc_request_for(
                            "bevy-surface-acquire",
                            "render.surface.acquire",
                            &session,
                            session.config.wgpu_endpoint,
                            json!({
                                "surface_id": session.surface_id,
                                "pid": std::process::id()
                            }),
                        );
                        let _ = transport.requests.send(TransportRequest {
                            endpoint: session.config.wgpu_endpoint,
                            request,
                            lane: RpcLane::WorldLatest,
                        });
                        if response.request_id.0 == "bevy-surface-open" {
                            session.acquire_requested = true;
                        }
                    }
                    if request_id.0 == "bevy-world-surface-open" {
                        let request = rpc_request_for(
                            "bevy-world-surface-acquire",
                            "render.surface.acquire",
                            &session,
                            session.config.wgpu_endpoint,
                            json!({"surface_id": WORLD_SURFACE_ID, "pid": std::process::id()}),
                        );
                        let _ = transport.requests.send(TransportRequest {
                            endpoint: session.config.wgpu_endpoint,
                            request,
                            lane: RpcLane::WorldLatest,
                        });
                    }
                    if request_id.0.starts_with("bevy-flow-source") {
                        if let Some(program_revision) = result.get("program_revision").cloned() {
                            if let Ok(revision) =
                                serde_json::from_value::<UiProgramRevision>(program_revision)
                            {
                                bridge.identity.program_revision = revision;
                                if let Ok(mut screen) = screens.single_mut() {
                                    screen.identity.program_revision =
                                        bridge.identity.program_revision.clone();
                                }
                                bridge.flow_ready = true;
                            }
                        }
                    }
                    if request_id.0.starts_with("bevy-surface-frame") {
                        if let Some(frame_sequence) = result
                            .get("frame_sequence")
                            .and_then(serde_json::Value::as_u64)
                        {
                            session.frame_sequence = session.frame_sequence.max(frame_sequence);
                        }
                        let frame_request_id = format!(
                            "bevy-surface-frame-{}",
                            monotonic_timestamp_ns()
                        );
                        let request = rpc_request_for(
                            frame_request_id,
                            "render.surface.frame",
                            &session,
                            session.config.wgpu_endpoint,
                            json!({"surface_id": session.surface_id}),
                        );
                        if session.config.service_mode == Neon3ServiceMode::AutoWindowed {
                            let _ = transport.requests.send(TransportRequest {
                                endpoint: session.config.wgpu_endpoint,
                                request,
                                lane: RpcLane::WorldLatest,
                            });
                        }
                    }
                    // The runtime owns the input revision. Advance only from an
                    // accepted response (any request kind: nui-flow input
                    // frames, renderer clicks, semantic intents all echo the
                    // authoritative revision) and never regress on a delayed
                    // response. Failing to sync after a click lets the next
                    // input frame fail with ui_program_stale_input_revision.
                    let next_revision = ["input_revision", "accepted_input_revision"]
                        .into_iter()
                        .find_map(|key| result.get(key).and_then(serde_json::Value::as_u64))
                        .or_else(|| {
                            result
                                .get("snapshot")
                                .and_then(|snapshot| snapshot.get("scalar_inputs"))
                                .and_then(|inputs| inputs.get("input_revision"))
                                .and_then(serde_json::Value::as_u64)
                        });
                    if let Some(next_revision) = next_revision {
                        bridge.identity.expected_input_revision.0 =
                            bridge.identity.expected_input_revision.0.max(next_revision);
                        if let Ok(mut screen) = screens.single_mut() {
                            screen.identity.expected_input_revision =
                                bridge.identity.expected_input_revision;
                        }
                    }
                    if request_id.0 == "bevy-surface-acquire"
                        || request_id.0 == "bevy-world-surface-acquire"
                    {
                        let parsed = result
                            .get("buffers")
                            .and_then(serde_json::Value::as_array)
                            .map(|buffers| {
                                buffers
                                    .iter()
                                    .filter_map(|buffer| {
                                        Some(Neon3ExternalSurfaceBufferHandle {
                                            buffer_index: buffer.get("buffer_index")?.as_u64()?
                                                as u32,
                                            color_texture_handle: buffer
                                                .get("color_texture_handle")?
                                                .as_u64()?
                                                as usize,
                                            color_fence_handle: buffer
                                                .get("color_fence_handle")?
                                                .as_u64()?
                                                as usize,
                                            consumer_release_fence_handle: buffer
                                                .get("consumer_release_fence_handle")?
                                                .as_u64()?
                                                as usize,
                                            depth_texture_handle: buffer
                                                .get("depth_texture_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                            depth_fence_handle: buffer
                                                .get("depth_fence_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                            depth_consumer_release_fence_handle: buffer
                                                .get("depth_consumer_release_fence_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        if response.request_id.0 == "bevy-world-surface-acquire" {
                            world_handles.buffers = parsed;
                            session.world_surface_acquired = !world_handles.buffers.is_empty();
                            continue;
                        }
                        handles.buffers = result
                            .get("buffers")
                            .and_then(serde_json::Value::as_array)
                            .map(|buffers| {
                                buffers
                                    .iter()
                                    .filter_map(|buffer| {
                                        Some(Neon3ExternalSurfaceBufferHandle {
                                            buffer_index: buffer.get("buffer_index")?.as_u64()?
                                                as u32,
                                            color_texture_handle: buffer
                                                .get("color_texture_handle")?
                                                .as_u64()?
                                                as usize,
                                            color_fence_handle: buffer
                                                .get("color_fence_handle")?
                                                .as_u64()?
                                                as usize,
                                            consumer_release_fence_handle: buffer
                                                .get("consumer_release_fence_handle")?
                                                .as_u64()?
                                                as usize,
                                            depth_texture_handle: buffer
                                                .get("depth_texture_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                            depth_fence_handle: buffer
                                                .get("depth_fence_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                            depth_consumer_release_fence_handle: buffer
                                                .get("depth_consumer_release_fence_handle")
                                                .and_then(serde_json::Value::as_u64)
                                                .map(|value| value as usize),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        handles.color_texture_handle = result
                            .get("texture_handle")
                            .and_then(serde_json::Value::as_u64)
                            .map(|v| v as usize);
                        handles.color_fence_handle = result
                            .get("fence_handle")
                            .and_then(serde_json::Value::as_u64)
                            .map(|v| v as usize);
                        // Rendering consumes the per-buffer handles. A top-level
                        // texture handle without ring buffers is not a usable
                        // acquisition and must not enable the publish/render path.
                        session.surface_acquired = !handles.buffers.is_empty();
                        if session.surface_acquired
                            && session.config.service_mode == Neon3ServiceMode::AutoWindowed
                        {
                            let request = rpc_request_for(
                                format!("bevy-surface-frame-{}", monotonic_timestamp_ns()),
                                "render.surface.frame",
                                &session,
                                session.config.wgpu_endpoint,
                                json!({"surface_id": session.surface_id}),
                            );
                            let _ = transport.requests.send(TransportRequest {
                                endpoint: session.config.wgpu_endpoint,
                                request,
                                lane: RpcLane::WorldLatest,
                            });
                        }
                    }
                }
            }
            Ok(response) => {
                if let Some(entity) = image_entity
                    && let Ok(mut image) = external_images.get_mut(entity)
                {
                    image.pending_fingerprint = None;
                    image.upload_in_flight = false;
                }
                if response.request_id.0.starts_with("bevy-camera-") {
                    if let Some(camera) = latest.camera.as_mut() {
                        camera.in_flight = false;
                        camera.dirty = true;
                    }
                }
                if response.request_id.0.starts_with("bevy-anchor-batch-") {
                    if let Some(anchors) = latest.anchors.as_mut() {
                        anchors.in_flight = false;
                        anchors.dirty = true;
                    }
                }
                let error = response.error;
                let code = error.as_ref().map(|error| error.code.clone());
                if response.request_id.0.starts_with("bevy-pointer-") {
                    eprintln!(
                        "{}",
                        json!({
                            "event": "bevy_pointer_response",
                            "request_id": response.request_id.0,
                            "lane": format!("{lane:?}"),
                            "accepted": false,
                            "error_code": code,
                            "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                        })
                    );
                }
                if !response.request_id.0.starts_with("bevy-pointer-") {
                    warn!(
                        "Neon: request {} rejected: {:?}",
                        response.request_id.0, code
                    );
                    if let Some(ref error) = error {
                        warn!(
                            "Neon: request {} detail: {}",
                            response.request_id.0, error.message
                        );
                    }
                }
                if response.request_id.0.starts_with("nui-flow:") {
                    bridge.input_pending = false;
                    if let Some(result) = response.result.as_ref() {
                        if let Some(program_revision) = result
                            .get("expected_program_revision")
                            .or_else(|| result.get("program_revision"))
                            .cloned()
                            .and_then(|value| {
                                serde_json::from_value::<UiProgramRevision>(value).ok()
                            })
                        {
                            bridge.identity.program_revision = program_revision.clone();
                            if let Ok(mut screen) = screens.single_mut() {
                                screen.identity.program_revision = program_revision;
                            }
                        }
                    }
                }
                // Update input revision from ANY response — pointer events,
                // input frames, and host responses all carry the current
                // revision.  Without this, the next input frame is rejected
                // with "ui_program_stale_input_revision" because the adapter
                // advanced the revision during the semantic event dispatch.
                if let Some(result) = response.result.as_ref() {
                    if let Some(input_revision) = result
                        .get("expected_input_revision")
                        .or_else(|| result.get("input_revision"))
                        .and_then(serde_json::Value::as_u64)
                    {
                        bridge.identity.expected_input_revision.0 = input_revision;
                        if let Ok(mut screen) = screens.single_mut() {
                            screen.identity.expected_input_revision = Revision(input_revision);
                        }
                    }
                }
                if let Some(current_revision) =
                    error.as_ref().and_then(|error| error.current_revision)
                {
                    bridge.identity.expected_input_revision = current_revision;
                    if let Ok(mut screen) = screens.single_mut() {
                        screen.identity.expected_input_revision = current_revision;
                    }
                }
                // The snapshot was optimistically marked as sent when it
                // entered the transport queue. Re-open it after rejection
                // so the next update can retry with the current revision.
                bridge.sent = None;
                session.world_ui_snapshot = None;
                if let Ok(mut screen) = screens.single_mut() {
                    screen.sent = None;
                }
                session.last_error = code;
            }
            Err(error) => {
                if let Some(entity) = image_entity
                    && let Ok(mut image) = external_images.get_mut(entity)
                {
                    image.pending_fingerprint = None;
                    image.upload_in_flight = false;
                }
                // A failed interaction request must not disturb world latest
                // state, and vice versa. The lane metadata preserves this
                // boundary even when the RPC call failed before a response.
                if lane == RpcLane::WorldLatest {
                    if request_id.0.starts_with("bevy-camera-") {
                        if let Some(camera) = latest.camera.as_mut() {
                            camera.in_flight = false;
                            camera.dirty = true;
                        }
                    }
                    if request_id.0.starts_with("bevy-anchor-batch-") {
                        if let Some(anchors) = latest.anchors.as_mut() {
                            anchors.in_flight = false;
                            anchors.dirty = true;
                        }
                    }
                }
                warn!("Neon: transport error: {error}");
                session.last_error = Some(error);
            }
        }
    }
}

fn publish_camera_snapshot(
    session: ResMut<Neon3Session>,
    mut latest: ResMut<LatestWorldSubmission>,
    cameras: Query<(&Transform, &Projection), With<Camera3d>>,
) {
    if !session.surface_acquired || !session.world_surface_acquired || !session.config.world_ui {
        return;
    }
    let Ok((transform, projection)) = cameras.single() else {
        return;
    };
    let (vertical_fov_radians, near, far) = match projection {
        Projection::Perspective(perspective) => {
            (perspective.fov, perspective.near, perspective.far)
        }
        _ => return,
    };
    let frame = CameraFrame {
        camera_id: CameraId("bevy.main.camera".into()),
        world_space_id: WorldSpaceId("case.bevy.world.main".into()),
        producer_epoch: session.producer_epoch,
        sequence: 0,
        timestamp_monotonic_ns: monotonic_timestamp_ns(),
        payload: CameraFramePayload::ThreeDimensional {
            position: [
                transform.translation.x as f64,
                transform.translation.y as f64,
                transform.translation.z as f64,
            ],
            orientation: transform.rotation.into(),
            vertical_fov_radians,
            near,
            far,
        },
    };
    let signature = camera_signature(&frame, 0, 0);
    let changed = latest
        .camera
        .as_ref()
        .map_or(true, |pending| !same_camera_frame(&pending.frame, &frame));
    if changed {
        let in_flight = latest
            .camera
            .as_ref()
            .is_some_and(|pending| pending.in_flight);
        latest.camera = Some(PendingCameraSubmission {
            frame,
            signature,
            dirty: true,
            in_flight,
        });
    }
}

fn publish_world_anchor(
    session: ResMut<Neon3Session>,
    mut latest: ResMut<LatestWorldSubmission>,
    world_uis: Query<(&Transform, &NeonWorldUi<CharacterStatusVars>)>,
    cameras: Query<(&GlobalTransform, &Projection), With<Camera3d>>,
) {
    if !session.surface_acquired || !session.world_surface_acquired || !session.config.world_ui {
        return;
    }
    let Ok((camera_transform, projection)) = cameras.single() else {
        return;
    };
    // Bevy's own projection is the single source of truth for world -> screen:
    // the runtime consumes the normalized placement below and never re-projects.
    let view_matrix = camera_transform.to_matrix().inverse();
    let (projection_matrix, near, far) = match projection {
        Projection::Perspective(perspective) => (
            // The same matrix Bevy's camera uses internally (reverse-Z
            // infinite perspective); consistent with the prepass depth.
            Mat4::perspective_infinite_reverse_rh(
                perspective.fov,
                perspective.aspect_ratio,
                perspective.near,
            ),
            perspective.near,
            perspective.far,
        ),
        _ => return,
    };
    let view_projection = projection_matrix * view_matrix;
    let mut placements = Vec::new();
    for (transform, world_ui) in &world_uis {
        let attach = transform.translation + Vec3::new(0.0, world_ui.offset.y, 0.0);
        let view_position = view_matrix * attach.extend(1.0);
        let clip = view_projection * attach.extend(1.0);
        // Scene depth is reconstructed from camera-space -Z. Use that same
        // quantity for the UI anchor instead of relying on clip.w's projection
        // convention, so both sides compare the identical view-space distance.
        let actual_view_distance = -view_position.z;
        let (screen_x, screen_y) = if actual_view_distance > near && actual_view_distance <= far {
            let ndc = clip.truncate() / actual_view_distance;
            if ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 {
                // Normalized viewport coords, y-down from the top-left.
                (ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5))
            } else {
                // Off-screen: sentinel hides the panel in the runtime.
                (-1.0, -1.0)
            }
        } else {
            // Behind the camera or outside [near, far]: hidden.
            (-1.0, -1.0)
        };
        placements.push(WorldUiAnchorSample {
            anchor_id: neon_world_bridge::WorldAnchorId(world_ui.anchor.clone()),
            position: [
                transform.translation.x as f64,
                attach.y as f64,
                transform.translation.z as f64,
            ],
            billboard: true,
            occlusion: world_ui.occlusion.as_str().into(),
            screen_x,
            screen_y,
            // Keep the real camera-space distance here. The runtime uses it
            // for scene occlusion; billboard visual size is kept fixed by the
            // renderer independently.
            view_distance: actual_view_distance,
        });
    }
    placements.sort_by(|left, right| left.anchor_id.cmp(&right.anchor_id));
    let batch = WorldUiAnchorBatch {
        world_space_id: WorldSpaceId("case.bevy.world.main".into()),
        producer_epoch: session.producer_epoch,
        sequence: 0,
        timestamp_monotonic_ns: monotonic_timestamp_ns(),
        anchors: placements,
    };
    let changed = latest
        .anchors
        .as_ref()
        .map_or(true, |pending| !same_anchor_batch(&pending.batch, &batch));
    if changed {
        let in_flight = latest
            .anchors
            .as_ref()
            .is_some_and(|pending| pending.in_flight);
        latest.anchors = Some(PendingAnchorSubmission {
            signature: WorldFrameSignature {
                camera_position: [0; 3],
                camera_orientation: [0; 4],
                fov: 0,
                near: 0,
                far: 0,
                anchor_count: batch.anchors.len() as u32,
                anchor_hash: anchor_hash(&batch),
            },
            batch,
            dirty: true,
            in_flight,
        });
    }
}

fn flush_world_latest(
    mut session: ResMut<Neon3Session>,
    mut latest: ResMut<LatestWorldSubmission>,
    transport: Res<Neon3Transport>,
) {
    let (Some(camera_frame), Some(anchor_batch)) = (
        latest.camera.as_ref().map(|pending| pending.frame.clone()),
        latest.anchors.as_ref().map(|pending| pending.batch.clone()),
    ) else {
        return;
    };
    let signature = world_frame_signature(&camera_frame, &anchor_batch);

    let same_as_last = latest.last_sent_signature == Some(signature)
        && latest
            .last_sent_camera
            .as_ref()
            .is_some_and(|last| same_camera_frame(last, &camera_frame))
        && latest
            .last_sent_anchors
            .as_ref()
            .is_some_and(|last| same_anchor_batch(last, &anchor_batch));
    if !same_as_last {
        session.world_frame_sequence = session.world_frame_sequence.saturating_add(1);
        let sequence = session.world_frame_sequence;
        let timestamp = monotonic_timestamp_ns();
        if let Some(camera) = latest.camera.as_mut() {
            camera.frame.sequence = sequence;
            camera.frame.timestamp_monotonic_ns = timestamp;
            camera.signature = signature;
            camera.dirty = true;
        }
        if let Some(anchors) = latest.anchors.as_mut() {
            anchors.batch.sequence = sequence;
            anchors.batch.timestamp_monotonic_ns = timestamp;
            anchors.signature = signature;
            anchors.dirty = true;
        }
        latest.last_sent_signature = Some(signature);
        latest.last_sent_camera = latest.camera.as_ref().map(|pending| pending.frame.clone());
        latest.last_sent_anchors = latest.anchors.as_ref().map(|pending| pending.batch.clone());
    }

    let send_camera = latest
        .camera
        .as_ref()
        .is_some_and(|camera| camera.dirty && !camera.in_flight);
    if send_camera {
        let camera = latest.camera.as_ref().expect("camera pending");
        let request = rpc_request_for(
            format!("bevy-camera-{}", camera.frame.sequence),
            "wgpu.world.camera.submit_frame",
            &session,
            session.config.wgpu_endpoint,
            json!(camera.frame),
        );
        if transport
            .requests
            .send(TransportRequest {
                endpoint: session.config.wgpu_endpoint,
                request,
                lane: RpcLane::WorldLatest,
            })
            .is_ok()
        {
            if let Some(camera) = latest.camera.as_mut() {
                camera.in_flight = true;
                camera.dirty = false;
            }
        }
    }
    let send_anchors = latest
        .anchors
        .as_ref()
        .is_some_and(|anchors| anchors.dirty && !anchors.in_flight);
    if send_anchors {
        let anchors = latest.anchors.as_ref().expect("anchors pending");
        let request = rpc_request_for(
            format!("bevy-anchor-batch-{}", anchors.batch.sequence),
            "wgpu.world.ui.anchor.submit_batch",
            &session,
            session.config.wgpu_endpoint,
            json!(anchors.batch),
        );
        if transport
            .requests
            .send(TransportRequest {
                endpoint: session.config.wgpu_endpoint,
                request,
                lane: RpcLane::WorldLatest,
            })
            .is_ok()
        {
            if let Some(anchors) = latest.anchors.as_mut() {
                anchors.in_flight = true;
                anchors.dirty = false;
            }
        }
    }
}

fn camera_signature(
    frame: &CameraFrame,
    anchor_count: u32,
    anchor_hash: u64,
) -> WorldFrameSignature {
    let CameraFramePayload::ThreeDimensional {
        position,
        orientation,
        vertical_fov_radians,
        near,
        far,
    } = &frame.payload
    else {
        return WorldFrameSignature {
            camera_position: [0; 3],
            camera_orientation: [0; 4],
            fov: 0,
            near: 0,
            far: 0,
            anchor_count,
            anchor_hash,
        };
    };
    WorldFrameSignature {
        camera_position: position.map(|value| (value as f32).to_bits()),
        camera_orientation: orientation.map(f32::to_bits),
        fov: vertical_fov_radians.to_bits(),
        near: near.to_bits(),
        far: far.to_bits(),
        anchor_count,
        anchor_hash,
    }
}

fn anchor_hash(batch: &WorldUiAnchorBatch) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut anchors = batch.anchors.iter().collect::<Vec<_>>();
    anchors.sort_by(|left, right| left.anchor_id.cmp(&right.anchor_id));
    for anchor in anchors {
        for byte in anchor.anchor_id.0.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for value in anchor.position {
            hash ^= value.to_bits();
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for value in [
            u64::from(anchor.billboard),
            u64::from(anchor.occlusion.len() as u32),
            u64::from(anchor.screen_x.to_bits()),
            u64::from(anchor.screen_y.to_bits()),
            u64::from(anchor.view_distance.to_bits()),
        ] {
            hash ^= value;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for byte in anchor.occlusion.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn world_frame_signature(
    camera: &CameraFrame,
    anchors: &WorldUiAnchorBatch,
) -> WorldFrameSignature {
    camera_signature(camera, anchors.anchors.len() as u32, anchor_hash(anchors))
}

fn same_camera_frame(left: &CameraFrame, right: &CameraFrame) -> bool {
    left.camera_id == right.camera_id
        && left.world_space_id == right.world_space_id
        && left.producer_epoch == right.producer_epoch
        && left.payload == right.payload
}

fn same_anchor_batch(left: &WorldUiAnchorBatch, right: &WorldUiAnchorBatch) -> bool {
    left.world_space_id == right.world_space_id
        && left.producer_epoch == right.producer_epoch
        && left.anchors == right.anchors
}

fn monotonic_timestamp_ns() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn mouse_button_name(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "primary".into(),
        MouseButton::Right => "secondary".into(),
        MouseButton::Middle => "auxiliary".into(),
        MouseButton::Back => "back".into(),
        MouseButton::Forward => "forward".into(),
        MouseButton::Other(value) => format!("other:{value}"),
    }
}

fn pressed_mouse_buttons(mouse: &ButtonInput<MouseButton>) -> Vec<String> {
    [
        MouseButton::Left,
        MouseButton::Right,
        MouseButton::Middle,
        MouseButton::Back,
        MouseButton::Forward,
    ]
    .into_iter()
    .filter(|button| mouse.pressed(*button))
    .map(mouse_button_name)
    .collect()
}

fn active_modifiers(keyboard: &ButtonInput<KeyCode>) -> Vec<String> {
    [
        (KeyCode::ShiftLeft, "shift"),
        (KeyCode::ControlLeft, "control"),
        (KeyCode::AltLeft, "alt"),
        (KeyCode::SuperLeft, "meta"),
    ]
    .into_iter()
    .filter(|(key, _)| keyboard.pressed(*key))
    .map(|(_, name)| name.into())
    .collect()
}

fn queue_pointer_event(
    queue: &mut Neon3PointerEvents,
    state: &mut Neon3PointerState,
    session: &Neon3Session,
    event_type: &str,
    pixel: Option<[u32; 2]>,
    delta: [f32; 2],
    delta_mode: &str,
    button: Option<String>,
    buttons: Vec<String>,
    modifiers: Vec<String>,
) {
    const MAX_PENDING_POINTER_EVENTS: usize = 2048;
    if queue.events.len() >= MAX_PENDING_POINTER_EVENTS {
        queue.events.remove(0);
    }
    state.next_sequence = state.next_sequence.saturating_add(1);
    queue.events.push(Neon3PointerEvent {
        event_type: event_type.into(),
        pixel,
        delta,
        delta_mode: delta_mode.into(),
        button,
        buttons,
        modifiers,
        pointer_id: 0,
        sequence: state.next_sequence,
        generation: session.generation,
        frame_sequence: session.frame_sequence,
    });
}

fn flush_pointer_events(
    mut queue: ResMut<Neon3PointerEvents>,
    session: Res<Neon3Session>,
    transport: Res<Neon3Transport>,
) {
    // The WGPU runtime's `external_pointer_event` / `pointer_event` resolve the
    // hit against the combined fragment set (screen UI + world UI via the ID
    // renderer).  The `surface_id` in the event is validated by the UI runtime's
    // host adapter but never selects a separate interaction tree — the comment
    // in the WGPU runtime says "Screen and World UI share one hit image."
    // Always send pointer events to the SCREEN surface so the physics console
    // (sliders / buttons) receives interaction.  The world surface hit target
    // is resolved from the same combined fragment set.
    let surface_id = session.surface_id.as_str();
    // Pointer moves are latest-value data. Collapse a queued run of moves so
    // slow IPC/RPC frames cannot make the cursor visibly lag behind the OS.
    let mut events = Vec::new();
    for event in queue.events.drain(..) {
        if event.event_type == "move"
            && events.last().is_some_and(|last: &Neon3PointerEvent| last.event_type == "move")
        {
            let _ = events.pop();
        }
        events.push(event);
    }
    for event in events {
        let request_id = format!("bevy-pointer-{}", event.sequence);
        let trace_event = matches!(event.event_type.as_str(), "down" | "up");
        if trace_event {
            eprintln!(
                "{}",
                json!({
                    "event": "bevy_pointer_send",
                    "request_id": request_id.clone(),
                    "event_type": event.event_type.clone(),
                    "surface_id": surface_id,
                    "pixel": event.pixel,
                    "sequence": event.sequence,
                    "frame_sequence": event.frame_sequence,
                    "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                })
            );
        }
        let request = rpc_request_for(
            request_id.clone(),
            // UI Runtime is the first semantic consumer. It synchronously
            // resolves the renderer hit, dispatches its state machine, and
            // submits the presentation motion before Bevy/domain code sees
            // any semantic result.
            "ui.host.inbound",
            &session,
            session.config.ui_endpoint,
            json!({
                "kind": "pointer_event",
                "event": {
                    "event_type": event.event_type,
                    "surface_id": surface_id,
                    "pixel": event.pixel.unwrap_or([0, 0]),
                    "delta": event.delta,
                    "delta_mode": event.delta_mode,
                    "button": event.button,
                    "buttons": event.buttons,
                    "modifiers": event.modifiers,
                    "pointer_id": event.pointer_id,
                    "sequence": event.sequence,
                    "generation": event.generation,
                    "frame_sequence": event.frame_sequence,
                    "timestamp_monotonic_ns": monotonic_timestamp_ns(),
                }
            }),
        );
        if transport
            .requests
            .send(TransportRequest {
                endpoint: session.config.ui_endpoint,
                request,
                lane: RpcLane::Interaction,
            })
            .is_err()
        {
            warn!("Neon UI pointer event queue is closed");
            break;
        }
    }
}

fn publish_pointer_events(
    session: Res<Neon3Session>,
    mut queue: ResMut<Neon3PointerEvents>,
    mut state: ResMut<Neon3PointerState>,
    mut wheel: MessageReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    if !session.surface_acquired {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let focused = window.focused;
        let current_pixel = window.cursor_position().map(|position| {
            let width = window.width().max(1.0);
            let height = window.height().max(1.0);
            // Neon's headless external interaction plan is in logical 1280x720
            // coordinates while the shared color surface is backed at 2x. The
            // render texture remains high resolution; pointer hit testing must
            // consume the logical coordinate pair.
            let logical_scale = UI_BACKING_SCALE as f32;
            [
                ((position.x / width) * session.config.surface_size[0] as f32 / logical_scale)
                    .floor()
                    .clamp(0.0, (session.config.surface_size[0] / UI_BACKING_SCALE).saturating_sub(1) as f32)
                    as u32,
                ((position.y / height) * session.config.surface_size[1] as f32 / logical_scale)
                    .floor()
                    .clamp(0.0, (session.config.surface_size[1] / UI_BACKING_SCALE).saturating_sub(1) as f32)
                    as u32,
            ]
    });
    let buttons = pressed_mouse_buttons(&mouse);
    let modifiers = active_modifiers(&keyboard);
    let last_pixel = state.last_pixel;

    if state.focused && !focused {
        queue_pointer_event(
            &mut queue,
            &mut state,
            &session,
            "cancel",
            last_pixel,
            [0.0, 0.0],
            "pixel",
            None,
            buttons.clone(),
            modifiers.clone(),
        );
    }
    state.focused = focused;
    if !focused {
        return;
    }

    if current_pixel.is_some() && !state.inside {
        state.inside = true;
        queue_pointer_event(
            &mut queue,
            &mut state,
            &session,
            "enter",
            current_pixel,
            [0.0, 0.0],
            "pixel",
            None,
            buttons.clone(),
            modifiers.clone(),
        );
    } else if current_pixel.is_none() && state.inside {
        state.inside = false;
        queue_pointer_event(
            &mut queue,
            &mut state,
            &session,
            "leave",
            last_pixel,
            [0.0, 0.0],
            "pixel",
            None,
            buttons.clone(),
            modifiers.clone(),
        );
    }

    if current_pixel != state.last_pixel {
        state.last_pixel = current_pixel;
        if let Some(pixel) = current_pixel {
            queue_pointer_event(
                &mut queue,
                &mut state,
                &session,
                "move",
                Some(pixel),
                [0.0, 0.0],
                "pixel",
                None,
                buttons.clone(),
                modifiers.clone(),
            );
        }
    }

    for button in [
        MouseButton::Left,
        MouseButton::Right,
        MouseButton::Middle,
        MouseButton::Back,
        MouseButton::Forward,
    ] {
        let button_name = mouse_button_name(button);
        if mouse.just_pressed(button) {
            queue_pointer_event(
                &mut queue,
                &mut state,
                &session,
                "down",
                current_pixel,
                [0.0, 0.0],
                "pixel",
                Some(button_name.clone()),
                buttons.clone(),
                modifiers.clone(),
            );
        }
        if mouse.just_released(button) {
            queue_pointer_event(
                &mut queue,
                &mut state,
                &session,
                "up",
                current_pixel,
                [0.0, 0.0],
                "pixel",
                Some(button_name),
                buttons.clone(),
                modifiers.clone(),
            );
        }
    }

    for event in wheel.read() {
        let (delta, delta_mode) = match event.unit {
            MouseScrollUnit::Line => ([event.x, event.y], "line"),
            MouseScrollUnit::Pixel => ([event.x, event.y], "pixel"),
        };
        queue_pointer_event(
            &mut queue,
            &mut state,
            &session,
            "wheel",
            current_pixel.or(last_pixel),
            delta,
            delta_mode,
            None,
            buttons.clone(),
            modifiers.clone(),
        );
    }
}

fn flush_intents(
    mut queue: ResMut<Neon3IntentQueue>,
    session: Res<Neon3Session>,
    transport: Res<Neon3Transport>,
    bridge: Res<CharacterStatusBridge>,
) {
    for intent in queue.events.drain(..) {
        let request_id = intent.request_id.clone();
        let request = rpc_request_for(
            request_id.clone(),
            "ui.host.inbound",
            &session,
            session.config.ui_endpoint,
            json!({
                "kind": "semantic_intent",
                "event": {
                    "event_id": request_id.clone(),
                    "kind": "activate",
                    "intent": intent.action,
                    "source_node_key": "bevy.pointer.click",
                    "payload": intent.params,
                    "program_revision": bridge.identity.program_revision.clone(),
                    "input_revision": bridge.identity.expected_input_revision.0,
                    "request_id": request_id.clone(),
                    "idempotency_key": format!("bevy-ui-intent:{}", intent.request_id)
                }
            }),
        );
        let _ = transport.requests.send(TransportRequest {
            endpoint: session.config.ui_endpoint,
            request,
            lane: RpcLane::Interaction,
        });
    }
}

fn flush_input_changes(
    mut changes: ResMut<Neon3InputChanges>,
    mut bridge: ResMut<CharacterStatusBridge>,
    session: Res<Neon3Session>,
    transport: Res<Neon3Transport>,
) {
    if !bridge.flow_ready || bridge.input_pending || changes.changes.is_empty() {
        return;
    }
    bridge.identity.request_sequence = bridge.identity.request_sequence.saturating_add(1);
    let pending = std::mem::take(&mut changes.changes);
    let frame = UiInputFrame {
        program_revision: bridge.identity.program_revision.clone(),
        expected_input_revision: bridge.identity.expected_input_revision,
        request_id: format!("nui-flow:host-changes:{}", bridge.identity.request_sequence),
        idempotency_key: format!("nui-flow:host-changes:{}", bridge.identity.request_sequence),
        changes: pending
            .iter()
            .map(|(key, value)| UiInputChange {
                key: key.clone(),
                value: value.clone(),
            })
            .collect(),
    };
    if transport
        .requests
        .send(TransportRequest {
            endpoint: session.config.ui_endpoint,
            request: rpc_request_for(
                frame.request_id.clone(),
                "ui.input.frame",
                &session,
                session.config.ui_endpoint,
                json!(frame),
            ),
            lane: RpcLane::Interaction,
        })
        .is_ok()
    {
        bridge.input_pending = true;
    } else {
        changes.changes.extend(pending);
    }
}

fn flush_character_status(
    mut bridge: ResMut<CharacterStatusBridge>,
    mut session: ResMut<Neon3Session>,
    transport: Res<Neon3Transport>,
    world_uis: Query<&NeonWorldUi<CharacterStatusVars>>,
    mut screens: Query<&mut NeonScreenUi<CharacterStatusVars>, With<CharacterStatusScreenKey>>,
) {
    if !bridge.flow_ready {
        return;
    }
    if bridge.input_pending {
        return;
    }
    let mut rows = Vec::with_capacity(world_uis.iter().count());
    let mut world_changed = false;
    let mut screen_pending = None;
    if session.config.world_ui {
        // The combined flow carries screen scalar inputs and one bounded input
        // slot per monster. Publish both kinds of changes in one input frame.
        for world_ui in &world_uis {
            let Some(index) = world_ui
                .anchor
                .strip_prefix("monster.m")
                .and_then(|value| value.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(mut row) = world_ui.row_snapshot() else {
                continue;
            };
            row.values = row
                .values
                .into_iter()
                .map(|(key, value)| (format!("{key}{index}"), value))
                .collect();
            rows.push(row);
        }
        rows.sort_by(|left, right| left.stable_row_key.cmp(&right.stable_row_key));
        world_changed = session.world_ui_snapshot.as_ref() != Some(&rows);
    }
    let mut changes = if world_changed {
        rows.iter()
            .flat_map(|row| {
                row.values
                    .iter()
                    .map(|(key, value)| neon_ui::UiInputChange {
                        key: key.clone(),
                        value: value.clone(),
                    })
            })
            .collect()
    } else {
        Vec::new()
    };
    if let Ok(mut screen) = screens.single_mut() {
        let current = screen.vars.clone();
        let sent = screen.sent.clone();
        let frame = match sent {
            None => Some(current.snapshot(&mut screen.identity)),
            Some(previous) => current.diff(&previous, &mut screen.identity),
        };
        if let Some(frame) = frame {
            changes.extend(frame.changes);
            screen_pending = Some(current);
        }
    }
    if changes.is_empty() {
        return;
    }
    bridge.identity.request_sequence = bridge.identity.request_sequence.saturating_add(1);
    let frame = UiInputFrame {
        program_revision: bridge.identity.program_revision.clone(),
        expected_input_revision: bridge.identity.expected_input_revision,
        request_id: format!("nui-flow:combined-ui:{}", bridge.identity.request_sequence),
        idempotency_key: format!("nui-flow:combined-ui:{}", bridge.identity.request_sequence),
        changes,
    };
    let request = rpc_request_for(
        frame.request_id.clone(),
        "ui.input.frame",
        &session,
        session.config.ui_endpoint,
        json!(frame),
    );
    if transport
        .requests
        .send(TransportRequest {
            endpoint: session.config.ui_endpoint,
            request,
            lane: RpcLane::Interaction,
        })
        .is_ok()
    {
        bridge.input_pending = true;
        if world_changed {
            session.world_ui_snapshot = Some(rows);
        }
        if let Some(current) = screen_pending {
            if let Ok(mut screen) = screens.single_mut() {
                screen.sent = Some(current);
            }
        }
    } else {
        bridge.input_pending = false;
    }
}

fn rpc_request_for(
    id: impl Into<String>,
    method: &str,
    session: &Neon3Session,
    _endpoint: SocketAddr,
    params: serde_json::Value,
) -> RpcRequest {
    let id = id.into();
    RpcRequest {
        protocol: neon_protocol::RPC_PROTOCOL.into(),
        version: PROTOCOL_VERSION,
        request_id: RequestId(id.clone()),
        client: ClientIdentity {
            kind: ClientKind::ExternalHost,
            instance_id: session.config.session_id.clone(),
            pid: std::process::id(),
            origin: "neon3-bevy-nui-host".into(),
        },
        target: ServiceName(if method.starts_with("ui.") {
            "ui-runtime".into()
        } else {
            "wgpu-runtime".into()
        }),
        method: method.into(),
        params,
        expected_revision: None,
        idempotency_key: Some(format!("bevy:{method}:{id}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_response_round_trip_extracts_physics_kick() {
        let result = json!({
            "state": "accepted",
            "semantic_intent": {
                "event_id": "wgpu-pointer-click-1",
                "intent": "phys.kick.12",
                "source_node_key": "phys-tag-12",
                "payload": {},
                "requested_value": null
            }
        });
        let event = semantic_intent_from_result(&result, "bevy-pointer-1")
            .expect("semantic response must reach the host event queue");
        assert_eq!(event.intent, "phys.kick.12");
        assert_eq!(event.source_node_key.as_deref(), Some("phys-tag-12"));
    }

    #[test]
    fn ui_cases_parse_as_declared() {
        let ordinary =
            neon_ui::parse_nui_flow(ORDINARY_STATUS_NUI).expect("ordinary UI case must parse");
        assert!(ordinary.world_panels.is_empty());

        let world = neon_ui::parse_nui_flow(include_str!("../assets/ui/character-status.nui"))
            .expect("world UI case must parse");
        assert_eq!(world.world_panels.len(), 1);
        assert_eq!(
            world.world_panels[0]
                .anchor_id
                .as_ref()
                .map(|id| id.0.as_str()),
            Some("player.main")
        );
    }

    #[test]
    fn bevy_image_source_accepts_rgba8_and_rejects_gpu_only_images() {
        use bevy::asset::RenderAssetUsages;
        use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

        let image = Image::new(
            Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD,
        );
        let source = neon3_image_source("test-image", &image).expect("RGBA8 source");
        assert_eq!(source.image_id, "test-image");
        assert_eq!((source.width, source.height), (2, 1));
        assert_eq!(source.bytes.len(), 8);

        let oversized = Image::new(
            Extent3d {
                width: 3000,
                height: 1500,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![64; 3000 * 1500 * 4],
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD,
        );
        let thumbnail = neon3_image_source("thumbnail", &oversized).expect("bounded RGBA8");
        assert_eq!((thumbnail.width, thumbnail.height), (2046, 1023));
        assert_eq!(thumbnail.bytes.len(), 2046 * 1023 * 4);

        let gpu_only = Image::new_uninit(
            image.texture_descriptor.size,
            TextureDimension::D2,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        assert!(neon3_image_source("gpu-only", &gpu_only).is_none());
    }

    #[test]
    fn flow_vars_generate_full_snapshot_and_sparse_diff() {
        let mut identity = NuiFlowIdentity {
            program_revision: UiProgramRevision {
                program_id: "character.player.main.status".into(),
                revision: Revision(3),
                schema_version: neon_ui::UI_PROGRAM_SCHEMA_VERSION,
                capabilities: Vec::new(),
            },
            expected_input_revision: Revision(7),
            request_sequence: 0,
        };
        let before = CharacterStatusVars {
            health: 82.0,
            mana: 64.0,
            level: 12,
        };
        let snapshot = before.snapshot(&mut identity);
        assert_eq!(snapshot.changes.len(), 3);
        let after = CharacterStatusVars {
            health: 76.0,
            ..before.clone()
        };
        let diff = after.diff(&before, &mut identity).expect("health changed");
        assert_eq!(diff.changes.len(), 1);
        assert_eq!(diff.changes[0].key, "health");
        assert!(after.diff(&after, &mut identity).is_none());
    }

    #[test]
    fn apply_changes_writes_back_typed_variables() {
        let mut vars = CharacterStatusVars {
            health: 82.0,
            mana: 64.0,
            level: 12,
        };
        let changes = vec![
            neon_ui::UiInputChange {
                key: "health".into(),
                value: neon_ui::UiInputValue::F32 { value: 31.0 },
            },
            neon_ui::UiInputChange {
                key: "level".into(),
                value: neon_ui::UiInputValue::U32 { value: 20 },
            },
        ];
        vars.apply_changes(&changes).expect("valid writeback");
        assert_eq!(vars.health, 31.0);
        assert_eq!(vars.level, 20);
        assert_eq!(vars.mana, 64.0);

        let wrong_type = vec![neon_ui::UiInputChange {
            key: "health".into(),
            value: neon_ui::UiInputValue::Bool { value: true },
        }];
        assert_eq!(
            vars.apply_changes(&wrong_type),
            Err(NuiFlowVarError::TypeMismatch("health".into()))
        );

        let unknown = vec![neon_ui::UiInputChange {
            key: "gold".into(),
            value: neon_ui::UiInputValue::U32 { value: 1 },
        }];
        assert_eq!(
            vars.apply_changes(&unknown),
            Err(NuiFlowVarError::UnknownKey("gold".into()))
        );
    }

    #[test]
    fn row_snapshot_maps_a_full_template_row() {
        let vars = CharacterStatusVars {
            health: 82.0,
            mana: 64.0,
            level: 12,
        };
        let row = vars.row_snapshot("npc.blacksmith");
        assert_eq!(row.stable_row_key, "npc.blacksmith");
        assert_eq!(row.values.len(), 3);
        assert_eq!(
            row.values["health"],
            neon_ui::UiInputValue::F32 { value: 82.0 }
        );
        assert_eq!(
            row.values["mana"],
            neon_ui::UiInputValue::F32 { value: 64.0 }
        );
        assert_eq!(
            row.values["level"],
            neon_ui::UiInputValue::U32 { value: 12 }
        );
    }

    #[test]
    fn generic_world_ui_uses_its_anchor_as_the_repeat_row_key() {
        let world_ui = NeonWorldUi::<CharacterStatusVars> {
            flow: "character-status".into(),
            vars: CharacterStatusVars {
                health: 82.0,
                mana: 64.0,
                level: 12,
            },
            identity: NuiFlowIdentity {
                program_revision: UiProgramRevision {
                    program_id: "character.player.main.status".into(),
                    revision: Revision(1),
                    schema_version: neon_ui::UI_PROGRAM_SCHEMA_VERSION,
                    capabilities: Vec::new(),
                },
                expected_input_revision: Revision(0),
                request_sequence: 0,
            },
            anchor: "npc.blacksmith".into(),
            offset: Vec3::Y,
            sent: None,
            visible: true,
            occlusion: WorldUiOcclusion::AlwaysVisible,
        };
        assert_eq!(world_ui.stable_row_key(), "npc.blacksmith");
        assert_eq!(
            world_ui.vars.row_snapshot(&world_ui.anchor).stable_row_key,
            "npc.blacksmith"
        );
        assert_eq!(
            world_ui.row_snapshot().expect("visible row").stable_row_key,
            "npc.blacksmith"
        );
    }

    #[test]
    fn world_frame_signature_is_order_independent_and_bit_exact() {
        let camera = CameraFrame {
            camera_id: CameraId("camera".into()),
            world_space_id: WorldSpaceId("world".into()),
            producer_epoch: 1,
            sequence: 10,
            timestamp_monotonic_ns: 100,
            payload: CameraFramePayload::ThreeDimensional {
                position: [1.25, 2.5, -3.75],
                orientation: [0.0, 0.5, 0.0, 1.0],
                vertical_fov_radians: 1.0,
                near: 0.1,
                far: 1000.0,
            },
        };
        let sample = |id: &str, x: f64| WorldUiAnchorSample {
            anchor_id: neon_world_bridge::WorldAnchorId(id.into()),
            position: [x, 1.0, 2.0],
            billboard: true,
            occlusion: "depth_tested".into(),
            screen_x: 0.5,
            screen_y: 0.5,
            view_distance: 4.0,
        };
        let first = WorldUiAnchorBatch {
            world_space_id: WorldSpaceId("world".into()),
            producer_epoch: 1,
            sequence: 10,
            timestamp_monotonic_ns: 100,
            anchors: vec![sample("b", 2.0), sample("a", 1.0)],
        };
        let second = WorldUiAnchorBatch {
            anchors: vec![sample("a", 1.0), sample("b", 2.0)],
            ..first.clone()
        };
        assert_eq!(
            world_frame_signature(&camera, &first),
            world_frame_signature(&camera, &second)
        );

        let mut changed = camera.clone();
        if let CameraFramePayload::ThreeDimensional { position, .. } = &mut changed.payload {
            position[0] = f32::from_bits(1.25_f32.to_bits() + 1) as f64;
        }
        assert_ne!(
            world_frame_signature(&camera, &first),
            world_frame_signature(&changed, &first)
        );
    }

    #[test]
    fn latest_submission_keeps_new_pending_value_while_old_request_is_in_flight() {
        let frame = CameraFrame {
            camera_id: CameraId("camera".into()),
            world_space_id: WorldSpaceId("world".into()),
            producer_epoch: 1,
            sequence: 1,
            timestamp_monotonic_ns: 1,
            payload: CameraFramePayload::ThreeDimensional {
                position: [0.0, 0.0, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                vertical_fov_radians: 1.0,
                near: 0.1,
                far: 1000.0,
            },
        };
        let mut latest = LatestWorldSubmission {
            camera: Some(PendingCameraSubmission {
                signature: camera_signature(&frame, 0, 0),
                frame: frame.clone(),
                dirty: false,
                in_flight: true,
            }),
            ..Default::default()
        };
        let mut next = frame;
        if let CameraFramePayload::ThreeDimensional { position, .. } = &mut next.payload {
            position[2] = 10.0;
        }
        let in_flight = latest
            .camera
            .as_ref()
            .is_some_and(|pending| pending.in_flight);
        latest.camera = Some(PendingCameraSubmission {
            signature: camera_signature(&next, 0, 0),
            frame: next.clone(),
            dirty: true,
            in_flight,
        });
        let pending = latest.camera.as_ref().expect("pending camera");
        assert!(pending.in_flight);
        assert!(pending.dirty);
        assert_eq!(pending.frame.payload, next.payload);
    }
}

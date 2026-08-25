//! Bevy-side D3D12 external texture import.
//!
//! This uses the wgpu version selected by Bevy 0.19.1 (29.0.4), independently
//! from Neon3's wgpu version. Only native D3D12 COM resources cross the process
//! boundary.

use std::fmt;

use windows::Win32::{
    Foundation::HANDLE,
    Graphics::Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource},
};

#[derive(Debug)]
pub enum ImportError {
    NotDx12,
    HalDeviceUnavailable,
    OpenSharedHandle(String),
    WaitExternalFence(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDx12 => write!(f, "Bevy RenderDevice is not using DX12"),
            Self::HalDeviceUnavailable => write!(f, "Bevy DX12 HAL device is unavailable"),
            Self::OpenSharedHandle(error) => write!(f, "open shared D3D12 texture: {error}"),
            Self::WaitExternalFence(error) => write!(f, "wait external D3D12 fence: {error}"),
        }
    }
}

impl std::error::Error for ImportError {}

#[derive(Debug)]
pub struct ImportedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub fence: ID3D12Fence,
    pub consumer_release_fence: ID3D12Fence,
}

pub fn import_texture(
    device: &wgpu::Device,
    handle: usize,
    fence_handle: usize,
    consumer_release_fence_handle: usize,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> Result<ImportedTexture, ImportError> {
    let hal_device = unsafe { device.as_hal::<wgpu_hal::api::Dx12>() }
        .ok_or(ImportError::HalDeviceUnavailable)?;
    let raw_device: &ID3D12Device = hal_device.raw_device();
    let raw_resource: ID3D12Resource = unsafe {
        let mut resource = None;
        raw_device
            .OpenSharedHandle(HANDLE(handle as *mut std::ffi::c_void), &mut resource)
            .map_err(|error| ImportError::OpenSharedHandle(error.to_string()))?;
        resource.ok_or_else(|| ImportError::OpenSharedHandle("null resource".into()))?
    };
    let fence: ID3D12Fence = unsafe {
        let mut fence = None;
        raw_device
            .OpenSharedHandle(HANDLE(fence_handle as *mut std::ffi::c_void), &mut fence)
            .map_err(|error| ImportError::OpenSharedHandle(error.to_string()))?;
        fence.ok_or_else(|| ImportError::OpenSharedHandle("null fence".into()))?
    };
    let consumer_release_fence: ID3D12Fence = unsafe {
        let mut fence = None;
        raw_device
            .OpenSharedHandle(
                HANDLE(consumer_release_fence_handle as *mut std::ffi::c_void),
                &mut fence,
            )
            .map_err(|error| ImportError::OpenSharedHandle(error.to_string()))?;
        fence.ok_or_else(|| ImportError::OpenSharedHandle("null consumer release fence".into()))?
    };
    let hal_texture = unsafe {
        wgpu_hal::dx12::Device::texture_from_raw(
            raw_resource,
            format,
            wgpu::TextureDimension::D2,
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            1,
            1,
        )
    };
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu_hal::api::Dx12>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("neon3-bevy-external-surface"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    };
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(ImportedTexture {
        texture,
        view,
        fence,
        consumer_release_fence,
    })
}

pub fn signal_consumer_release(
    queue: &wgpu::Queue,
    texture: &ImportedTexture,
    value: u64,
) -> Result<(), ImportError> {
    let hal_queue = unsafe { queue.as_hal::<wgpu_hal::api::Dx12>() }
        .ok_or(ImportError::HalDeviceUnavailable)?;
    unsafe {
        hal_queue
            .as_raw()
            .Signal(&texture.consumer_release_fence, value)
            .map_err(|error| ImportError::WaitExternalFence(error.to_string()))?;
    }
    Ok(())
}

pub fn completed_fence_value(texture: &ImportedTexture) -> u64 {
    unsafe { texture.fence.GetCompletedValue() }
}

/// Enqueue a GPU-side wait on Bevy's queue. Reading GetCompletedValue on the
/// CPU is not sufficient: the Bevy queue must wait before sampling the shared
/// resource, otherwise it can race Neon's render pass.
pub fn wait_external_fence(
    queue: &wgpu::Queue,
    texture: &ImportedTexture,
    value: u64,
) -> Result<(), ImportError> {
    let hal_queue = unsafe { queue.as_hal::<wgpu_hal::api::Dx12>() }
        .ok_or(ImportError::HalDeviceUnavailable)?;
    unsafe {
        hal_queue
            .as_raw()
            .Wait(&texture.fence, value)
            .map_err(|error| ImportError::WaitExternalFence(error.to_string()))?;
    }
    Ok(())
}

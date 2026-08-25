//! Generic asynchronous GPU readback helpers for diagnostics and probes.
//!
//! The implementation deliberately uses wgpu's `DownloadBuffer` instead of
//! keeping a second ad-hoc mapping implementation in each test tool.

use std::sync::mpsc::{self, Receiver};

use wgpu::util::DownloadBuffer;

#[derive(Debug)]
pub enum GpuReadbackError {
    InvalidLayout(&'static str),
    Map(String),
}

impl std::fmt::Display for GpuReadbackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLayout(message) => {
                write!(formatter, "invalid_gpu_readback_layout: {message}")
            }
            Self::Map(error) => write!(formatter, "gpu_readback_map_failed: {error}"),
        }
    }
}

impl std::error::Error for GpuReadbackError {}

pub type GpuReadbackReceiver = Receiver<Result<Vec<u8>, GpuReadbackError>>;

/// Downloads an existing GPU buffer without blocking the render thread.
pub fn download_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> GpuReadbackReceiver {
    let (sender, receiver) = mpsc::channel();
    DownloadBuffer::read_buffer(device, queue, &buffer.slice(..), move |result| {
        let result = result
            .map(|download| download.to_vec())
            .map_err(|error| GpuReadbackError::Map(error.to_string()));
        let _ = sender.send(result);
    });
    receiver
}

/// Copies a texture region into a staging buffer and downloads the bytes.
/// `bytes_per_row` must include wgpu's required 256-byte row alignment.
pub fn download_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    extent: wgpu::Extent3d,
    bytes_per_row: u32,
) -> Result<GpuReadbackReceiver, GpuReadbackError> {
    if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
        return Err(GpuReadbackError::InvalidLayout("texture extent is empty"));
    }
    let required_row_bytes = texture
        .format()
        .block_copy_size(None)
        .ok_or(GpuReadbackError::InvalidLayout(
            "texture format has no byte size",
        ))?
        .checked_mul(extent.width)
        .ok_or(GpuReadbackError::InvalidLayout("row byte count overflow"))?;
    if (bytes_per_row as usize) % (wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize) != 0
        || (bytes_per_row as usize) < required_row_bytes as usize
    {
        return Err(GpuReadbackError::InvalidLayout(
            "bytes_per_row is too small or not 256-byte aligned",
        ));
    }
    let size = (bytes_per_row as u64)
        .checked_mul(extent.height as u64)
        .and_then(|size| size.checked_mul(extent.depth_or_array_layers as u64))
        .ok_or(GpuReadbackError::InvalidLayout("staging size overflow"))?;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neon3-gpu-readback-staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("neon3-gpu-readback-copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(extent.height),
            },
        },
        extent,
    );
    queue.submit(Some(encoder.finish()));
    Ok(download_buffer(device, queue, &staging))
}

pub const fn aligned_bytes_per_row(unpadded_bytes_per_row: u32) -> u32 {
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded_bytes_per_row + alignment - 1) / alignment * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downloads_r32_float_texture_bytes() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
            ..Default::default()
        }))
        .expect("a headless adapter is required");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("neon3-gpu-readback-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("a device is required");
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("neon3-gpu-readback-test-texture"),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let values = [0.125f32, 0.875f32];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row(8)),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let receiver = download_texture(
            &device,
            &queue,
            &texture,
            wgpu::Origin3d::ZERO,
            wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            aligned_bytes_per_row(8),
        )
        .expect("texture readback should be scheduled");
        let bytes = loop {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Ok(result) = receiver.try_recv() {
                break result.expect("texture readback should complete");
            }
        };
        let downloaded = bytemuck::cast_slice::<u8, f32>(&bytes[..8]);
        assert_eq!(downloaded, &values);
    }
}

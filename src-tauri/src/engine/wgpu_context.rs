//! GPU adapter selection, device creation, and capability probing for the wgpu engine.

use anyhow::{Context, Result};
use tracing::info;
use wgpu::{Device, DeviceDescriptor, Features, Limits, Queue};

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// True when the adapter supports native f16 shader ops (shader_f16 feature).
    #[allow(dead_code)]
    pub f16_supported: bool,
    /// Usable VRAM ceiling from adapter limits (bytes). 0 = unknown.
    #[allow(dead_code)]
    pub max_buf_bytes: u64,
}

impl GpuContext {
    /// Attempt to initialise a GPU context on the best available adapter.
    /// Returns `None` when no suitable adapter is found (headless or no Vulkan/Metal/DX12).
    pub async fn try_init() -> Option<Self> {
        match Self::init_inner().await {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                tracing::warn!("wgpu init failed, will fall back to CPU: {:#}", e);
                None
            }
        }
    }

    async fn init_inner() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan, Metal, DX12
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("no GPU adapter found")?;

        let adapter_info = adapter.get_info();
        info!(
            "GPU adapter: {} ({:?}) backend={:?}",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );

        let f16_supported = adapter.features().contains(Features::SHADER_F16);
        let limits = adapter.limits();
        let max_buf_bytes = limits.max_storage_buffer_binding_size as u64;

        info!(
            "Adapter caps: f16={} max_storage_buffer={}MiB",
            f16_supported,
            max_buf_bytes / (1024 * 1024)
        );

        let mut required_features = Features::empty();
        if f16_supported {
            required_features |= Features::SHADER_F16;
        }

        // Request the adapter's actual limits so large tensor buffers (>128 MB) are allowed.
        // Limits::default() caps max_storage_buffer_binding_size at 128 MB and
        // max_buffer_size at 256 MB, which is too small for embedding tables and
        // large-vocab or F16 weights.
        let required_limits = Limits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            ..Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("tpt-spark"),
                    required_features,
                    required_limits,
                },
                None,
            )
            .await
            .context("failed to acquire device")?;

        Ok(Self {
            device,
            queue,
            adapter_info,
            f16_supported,
            max_buf_bytes,
        })
    }

    /// Create an uninitialised storage buffer of `size` bytes.
    pub fn create_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a host-readable buffer (MAP_READ | COPY_DST) for readback validation.
    pub fn create_readback_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

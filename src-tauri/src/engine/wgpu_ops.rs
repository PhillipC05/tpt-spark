//! Compiled wgpu compute pipelines and dispatch helpers for the inference loop.
//!
//! Pipelines are compiled once at model-load time and reused across all infer() calls.
//! Bind groups are created fresh per dispatch (they are cheap; pipelines are not).

use anyhow::Result;
use bytemuck::Pod;
use wgpu::{ComputePipeline, Device};

use crate::engine::wgpu_context::GpuContext;

// ── Shader sources embedded at compile time ───────────────────────────────────

const SRC_DEQUANT_Q8 : &str = include_str!("shaders/dequant_q8.wgsl");
const SRC_DEQUANT_Q4K: &str = include_str!("shaders/dequant_q4k.wgsl");
const SRC_GEMM       : &str = include_str!("shaders/gemm.wgsl");
const SRC_RMS_NORM   : &str = include_str!("shaders/rms_norm.wgsl");
const SRC_ROPE       : &str = include_str!("shaders/rope.wgsl");
const SRC_ATTENTION  : &str = include_str!("shaders/attention.wgsl");
const SRC_SILU       : &str = include_str!("shaders/silu.wgsl");

// ── Pipeline collection ────────────────────────────────────────────────────────

pub struct WgpuPipelines {
    pub dequant_q8     : ComputePipeline,
    pub dequant_q4k    : ComputePipeline,
    pub gemm           : ComputePipeline,
    pub rms_norm       : ComputePipeline,
    pub rope           : ComputePipeline,
    pub attn_scores    : ComputePipeline,
    pub attn_softmax   : ComputePipeline,
    pub attn_output    : ComputePipeline,
    pub silu           : ComputePipeline,
}

impl WgpuPipelines {
    pub fn compile(ctx: &GpuContext) -> Result<Self> {
        Ok(Self {
            dequant_q8  : make_pipeline(&ctx.device, "dequant_q8",   SRC_DEQUANT_Q8,  "main"),
            dequant_q4k : make_pipeline(&ctx.device, "dequant_q4k",  SRC_DEQUANT_Q4K, "main"),
            gemm        : make_pipeline(&ctx.device, "gemm",         SRC_GEMM,         "main"),
            rms_norm    : make_pipeline(&ctx.device, "rms_norm",     SRC_RMS_NORM,     "main"),
            rope        : make_pipeline(&ctx.device, "rope",         SRC_ROPE,         "main"),
            attn_scores : make_pipeline(&ctx.device, "attn_scores",  SRC_ATTENTION,    "attention_scores"),
            attn_softmax: make_pipeline(&ctx.device, "attn_softmax", SRC_ATTENTION,    "attention_softmax"),
            attn_output : make_pipeline(&ctx.device, "attn_output",  SRC_ATTENTION,    "attention_output"),
            silu        : make_pipeline(&ctx.device, "silu",         SRC_SILU,         "main"),
        })
    }
}

// ── Dispatch helpers ───────────────────────────────────────────────────────────

/// Dispatch `pipeline` with a fresh bind group built from `bindings`.
/// `bindings` is an ordered slice of (binding index, buffer, size_bytes or 0 for whole buffer).
pub fn dispatch(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    bindings: &[BindingEntry<'_>],
    uniform_data: Option<&dyn UniformBytes>,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
) {
    let device = &ctx.device;
    let queue  = &ctx.queue;

    // Upload uniform data if provided.
    let uniform_buf = uniform_data.map(|u| {
        let bytes = u.as_bytes();
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    });

    let bgl = pipeline.get_bind_group_layout(0);

    let mut entries: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|b| wgpu::BindGroupEntry {
            binding: b.binding,
            resource: b.buffer.as_entire_binding(),
        })
        .collect();

    if let Some(ref ub) = uniform_buf {
        entries.push(wgpu::BindGroupEntry {
            binding: entries.len() as u32,
            resource: ub.as_entire_binding(),
        });
    }

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &entries,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

pub struct BindingEntry<'a> {
    pub binding: u32,
    pub buffer: &'a wgpu::Buffer,
}

// ── Uniform helper ─────────────────────────────────────────────────────────────

pub trait UniformBytes {
    fn as_bytes(&self) -> &[u8];
}

impl<T: Pod> UniformBytes for T {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ── GPU readback (for validation) ─────────────────────────────────────────────

pub fn readback_f32(ctx: &GpuContext, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let rb = ctx.create_readback_buffer("readback", (n * 4) as u64);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &rb, 0, (n * 4) as u64);
    ctx.queue.submit([encoder.finish()]);

    let slice = rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    bytemuck::cast_slice::<u8, f32>(&data).to_vec()
}

// ── Internal helpers ───────────────────────────────────────────────────────────

fn make_pipeline(device: &Device, label: &str, src: &str, entry: &str) -> ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: entry,
        compilation_options: Default::default(),
    })
}

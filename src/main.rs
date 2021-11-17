use std::iter;
use std::path::Path;
use wgpu::util::*;
use wgpu::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

type Rgba = [f32; 4];
type RgbaData = [u8; 16];

struct Entity {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    constant_buffer: Buffer,
    constant_stage: Buffer, // Only used as staging buffer, i.e. when the sample app is run with the -s command line flag.
    bind_group: BindGroup,
    num_indices: u32,
    constant_data: RgbaData,
}

fn create_device(window: &Window, trace_path: Option<&Path>) -> (Surface, Adapter, Device, Queue) {
    let backends = Backends::all();
    let instance = Instance::new(backends);
    let surface = unsafe { instance.create_surface(window) };
    let adapter = select_adapter(&instance, &surface, backends);
    //let trace_path = None;
    let (device, queue) = futures::executor::block_on(adapter.request_device(
        &DeviceDescriptor {
            label: None,
            features: Features::empty(),
            limits: Limits::default(),
        },
        trace_path,
    ))
    .unwrap();
    (surface, adapter, device, queue)
}

struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
    entities: Vec<Entity>,
}

impl State {
    fn new(window: &mut Window, trace_path: Option<&Path>) -> Self {
        let size = window.inner_size();
        let (surface, adapter, device, queue) = create_device(window, trace_path);
        let info = adapter.get_info();
        window.set_title(&format!(
            "{} - {:?} - {:?}",
            info.name, info.backend, info.device_type
        ));
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VertexBufferLayout {
                    array_stride: 3 * std::mem::size_of::<f32>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState {
                        color: BlendComponent::REPLACE,
                        alpha: BlendComponent::REPLACE,
                    }),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                clamp_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });
        let entities = create_quads(&device, &bind_group_layout);
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            entities,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    async fn upload_constant_data_via_staging_buffers(&mut self) {
        // Copy constant data into staging buffers.
        {
            let buffer_slices: Vec<_> = self
                .entities
                .iter()
                .map(|e| e.constant_stage.slice(..))
                .collect();
            let mappings: Vec<_> = buffer_slices
                .iter()
                .map(|s| s.map_async(MapMode::Write))
                .collect();
            self.device.poll(Maintain::Wait);
            futures::future::join_all(mappings).await;
            for (slice, e) in buffer_slices.iter().zip(self.entities.iter()) {
                let mut view = slice.get_mapped_range_mut();
                view.as_mut().copy_from_slice(&e.constant_data);
            }
        }
        for e in &self.entities {
            e.constant_stage.unmap();
        }

        // Copy staging buffers to actual constant buffers.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Upload Encoder"),
            });
        const N: u64 = std::mem::size_of::<RgbaData>() as u64;
        for e in &self.entities {
            encoder.copy_buffer_to_buffer(&e.constant_stage, 0, &e.constant_buffer, 0, N);
        }

        self.queue.submit(iter::once(encoder.finish()));
    }

    fn render(&mut self, use_staging_buffers: bool) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if use_staging_buffers {
            futures::executor::block_on(self.upload_constant_data_via_staging_buffers());
        } else {
            for e in &self.entities {
                self.queue
                    .write_buffer(&e.constant_buffer, 0, &e.constant_data);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);

            for e in &self.entities {
                render_pass.set_vertex_buffer(0, e.vertex_buffer.slice(..));
                render_pass.set_index_buffer(e.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &e.bind_group, &[]);
                render_pass.draw_indexed(0..e.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let args = std::env::args().into_iter().collect();
    let trace_path = get_api_trace_path(&args);
    let use_staging = args.iter().any(|f| f == "-s" || f == "--staging");
    let event_loop = EventLoop::new();
    let size = winit::dpi::Size::Physical(winit::dpi::PhysicalSize {
        width: 512,
        height: 512,
    });
    let mut window = WindowBuilder::new()
        .with_inner_size(size)
        .build(&event_loop)
        .unwrap();
    let mut state = State::new(&mut window, trace_path);
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            _ => {}
        },
        Event::RedrawRequested(_) => {
            match state.render(use_staging) {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            };
        }
        _ => {}
    });
}

fn select_adapter(instance: &Instance, surface: &Surface, backends: Backends) -> Adapter {
    let mut adapters: Vec<Adapter> = instance
        .enumerate_adapters(backends)
        .filter(|adapter| adapter.is_surface_supported(&surface))
        .collect();
    if adapters.is_empty() {
        panic!("Failed to initialize adapter");
    }
    let mut selected = adapters.len();
    while selected >= adapters.len() {
        println!("Select adapter:");
        pretty_print(&adapters);
        println!("adapter index + enter:");
        selected = read_number().unwrap_or(adapters.len());
        if selected >= adapters.len() {
            println!("\nInvalid adapter index!\n");
        }
    }
    adapters.swap_remove(selected)
}

impl Entity {
    fn quad(
        device: &Device,
        layout: &BindGroupLayout,
        u0: f32,
        v0: f32,
        du: f32,
        dv: f32,
        color: Rgba,
    ) -> Self {
        let z = 0.5;
        #[rustfmt::skip]
        let positions = [
            u0,      v0,      z, // 0: bottom-left
            u0 + du, v0,      z, // 1: bottom-right
            u0,      v0 + dv, z, // 2: top-left
            u0 + du, v0 + dv, z, // 3: top-right
        ];
        let indices = [0, 1, 2, 2, 1, 3];
        let num_indices = indices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Entity Vertex Buffer"),
            contents: f32_slice_as_u8_slice(&positions),
            usage: BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Entity Index Buffer"),
            contents: u32_slice_as_u8_slice(&indices),
            usage: BufferUsages::INDEX,
        });
        let constant_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Entity Constant Buffer"),
            size: std::mem::size_of::<Rgba>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let constant_stage = device.create_buffer(&BufferDescriptor {
            label: Some("Entity Constant Buffer"),
            size: std::mem::size_of::<Rgba>() as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Entity Bind Group"),
            layout: layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &constant_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });
        Self {
            vertex_buffer,
            index_buffer,
            constant_buffer,
            constant_stage,
            bind_group,
            num_indices,
            constant_data: rgba_to_data(&color),
        }
    }
}

fn create_quads(device: &Device, bind_group_layout: &BindGroupLayout) -> Vec<Entity> {
    let colors = [
        [1.0, 0.0, 0.0, 1.0 as f32],
        [0.0, 1.0, 0.0, 1.0 as f32],
        [0.0, 0.0, 1.0, 1.0 as f32],
        [0.0, 1.0, 1.0, 1.0 as f32],
        [1.0, 0.0, 1.0, 1.0 as f32],
        [1.0, 1.0, 0.0, 1.0 as f32],
    ];
    let mut entities = Vec::new();
    const N: usize = 10;
    let du = 2.0 / N as f32;
    let dv = 2.0 / N as f32;
    let pad = 0.025;
    for v in 0..N {
        let v0 = -1.0 + v as f32 * dv;
        for u in 0..N {
            let u0 = -1.0 + u as f32 * du;
            let index = v * N + u;
            let color = colors[index % colors.len()];
            entities.push(Entity::quad(
                &device,
                &bind_group_layout,
                u0 + pad,
                v0 + pad,
                du - 2.0 * pad,
                dv - 2.0 * pad,
                color,
            ));
        }
    }
    entities
}

fn f32_slice_as_u8_slice(p: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            p.as_ptr() as *const u8,
            std::mem::size_of::<f32>() * p.len(),
        )
    }
}

fn u32_slice_as_u8_slice(p: &[u32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            p.as_ptr() as *const u8,
            std::mem::size_of::<u32>() * p.len(),
        )
    }
}

fn rgba_to_data(rgba: &Rgba) -> RgbaData {
    let rgba = f32_slice_as_u8_slice(rgba);
    const N: usize = std::mem::size_of::<RgbaData>();
    assert!(rgba.len() == N);
    let mut data = [0; N];
    for (i, &b) in rgba.iter().enumerate() {
        data[i] = b;
    }
    data
}

fn pad(length: usize, s: &str) -> String {
    let mut s = String::from(s);
    while s.len() < length {
        s.push(' ');
    }
    s
}

fn pretty_print(adapters: &[Adapter]) {
    let mut names = Vec::with_capacity(adapters.len());
    let mut backends = Vec::with_capacity(adapters.len());
    let mut device_types = Vec::with_capacity(adapters.len());
    for adapter in adapters {
        let info = adapter.get_info();
        names.push(info.name);
        backends.push(format!("{:?}", info.backend));
        device_types.push(format!("{:?}", info.device_type));
    }
    let name_length = names.iter().fold(0, |acc, x| acc.max(x.len()));
    let backend_length = backends.iter().fold(0, |acc, x| acc.max(x.len()));
    let device_type_length = device_types.iter().fold(0, |acc, x| acc.max(x.len()));
    for (i, ((name, backend), device_type)) in names
        .iter()
        .zip(backends.iter())
        .zip(device_types.iter())
        .enumerate()
    {
        println!(
            "  [{}] {} - {} - {}",
            i,
            pad(name_length, name),
            pad(backend_length, backend),
            pad(device_type_length, device_type)
        );
    }
}

fn read_number() -> Option<usize> {
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer).ok()?;
    buffer.trim().parse().ok()
}

fn get_api_trace_path<'a>(args: &'a Vec<String>) -> Option<&'a Path> {
    for (flag, dir) in args.iter().zip(args.iter().skip(1)) {
        if flag == "-t" || flag == "--trace" {
            let p = Path::new(dir);
            if p.exists() {
                return Some(p);
            } else {
                println!(
                    "WARNING [-t | --trace] : '{}' does not exist and will be ignored.",
                    dir
                );
            }
        }
    }
    None
}

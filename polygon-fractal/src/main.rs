mod geom;
mod render;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use eframe::egui;
use geom::{equalize, generate, set_side_count, ColourMode, FractalParams};
use render::{render_pixmap, straight_rgba, write_png};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Polygon Fractal",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

struct PreviewRequest {
    id: u64,
    params: FractalParams,
    width: u32,
    height: u32,
}

struct PreviewDone {
    id: u64,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    segments: usize,
    truncated: bool,
    elapsed_ms: f32,
}

struct ExportRequest {
    params: FractalParams,
    width: u32,
    height: u32,
    path: PathBuf,
}

struct ExportDone {
    path: PathBuf,
    elapsed_ms: f32,
    error: Option<String>,
}

struct App {
    params: FractalParams,
    export_width: u32,
    export_height: u32,
    preview_tx: Sender<PreviewRequest>,
    preview_rx: Receiver<PreviewDone>,
    export_tx: Sender<ExportRequest>,
    export_rx: Receiver<ExportDone>,
    preview_id: u64,
    panel_size: (u32, u32),
    texture: Option<egui::TextureHandle>,
    segment_count: usize,
    truncated: bool,
    preview_ms: f32,
    busy: bool,
    exporting: bool,
    export_status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let (preview_tx, preview_rx_worker) = mpsc::channel::<PreviewRequest>();
        let (preview_tx_worker, preview_rx) = mpsc::channel::<PreviewDone>();
        thread::spawn(move || preview_worker(preview_rx_worker, preview_tx_worker));

        let (export_tx, export_rx_worker) = mpsc::channel::<ExportRequest>();
        let (export_tx_worker, export_rx) = mpsc::channel::<ExportDone>();
        thread::spawn(move || export_worker(export_rx_worker, export_tx_worker));

        Self {
            params: FractalParams::default(),
            export_width: 4096,
            export_height: 4096,
            preview_tx,
            preview_rx,
            export_tx,
            export_rx,
            preview_id: 0,
            panel_size: (0, 0),
            texture: None,
            segment_count: 0,
            truncated: false,
            preview_ms: 0.0,
            busy: false,
            exporting: false,
            export_status: String::new(),
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let mut sides = self.params.side_lengths.len() as u32;
        if ui
            .add(egui::Slider::new(&mut sides, 3..=16).text("Sides"))
            .changed()
        {
            set_side_count(&mut self.params.side_lengths, sides as usize);
        }

        for (index, length) in self.params.side_lengths.iter_mut().enumerate() {
            ui.add(
                egui::Slider::new(length, 0.05..=4.0).text(format!("Side {}", index + 1)),
            );
        }
        if ui.button("Equalize").clicked() {
            equalize(&mut self.params.side_lengths);
        }

        ui.add(egui::Slider::new(&mut self.params.shrink, 0.05..=0.95).text("Shrink"));
        ui.add(egui::Slider::new(&mut self.params.depth, 1..=12).text("Depth"));
        ui.add(
            egui::Slider::new(&mut self.params.stroke_fraction, 0.002..=0.08).text("Stroke"),
        );
        ui.add(
            egui::Slider::new(&mut self.params.rotation_deg, -180.0..=180.0).text("Rotation"),
        );
        ui.add(egui::Slider::new(&mut self.params.twist_deg, -180.0..=180.0).text("Twist"));

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.inward, false, "Outward");
            ui.selectable_value(&mut self.params.inward, true, "Inward");
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Single, "Single");
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Gradient, "Gradient");
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Random, "Random");
        });
        match self.params.colour_mode {
            ColourMode::Single => {
                ui.horizontal(|ui| {
                    ui.label("Line");
                    ui.color_edit_button_srgb(&mut self.params.color_start);
                });
            }
            ColourMode::Gradient => {
                ui.horizontal(|ui| {
                    ui.label("Start");
                    ui.color_edit_button_srgb(&mut self.params.color_start);
                    ui.label("End");
                    ui.color_edit_button_srgb(&mut self.params.color_end);
                });
            }
            ColourMode::Random => {
                ui.label("Each line gets its own colour.");
            }
        }
        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_srgb(&mut self.params.background);
        });

        if ui
            .add_enabled(!self.busy, egui::Button::new("Update preview"))
            .on_hover_text("Draw the preview with the current settings")
            .clicked()
        {
            self.queue_preview();
        }

        let gap = geom::close_seed(&self.params.side_lengths).gap_ratio;
        ui.separator();
        ui.label(format!("Closure gap: {gap:.2}% of perimeter", gap = gap * 100.0));
        ui.label(format!("Segments: {}", self.segment_count));
        ui.label(format!("Preview: {:.0} ms", self.preview_ms));
        if self.truncated {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Segment limit reached. Lower the depth or shrink for a complete figure.",
            );
        }
        if self.busy {
            ui.label("Rendering preview…");
        }

        ui.separator();
        ui.label("Export size");
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.export_width)
                    .range(16..=16384)
                    .speed(16.0),
            );
            ui.label("×");
            ui.add(
                egui::DragValue::new(&mut self.export_height)
                    .range(16..=16384)
                    .speed(16.0),
            );
        });
        let export = ui
            .add_enabled(!self.exporting, egui::Button::new("Export PNG"))
            .on_hover_text("Choose a path and write an anti-aliased PNG");
        if export.clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Export fractal")
                .set_file_name("fractal.png")
                .add_filter("PNG image", &["png"])
                .save_file()
            {
                self.exporting = true;
                self.export_status = "Exporting…".to_string();
                let _ = self.export_tx.send(ExportRequest {
                    params: self.params.clone(),
                    width: self.export_width.clamp(16, 16384),
                    height: self.export_height.clamp(16, 16384),
                    path,
                });
            }
        }
        if !self.export_status.is_empty() {
            ui.label(&self.export_status);
        }
    }

    fn queue_preview(&mut self) {
        let (width, height) = self.panel_size;
        if width < 32 || height < 32 {
            return;
        }
        if self.params.colour_mode == ColourMode::Random {
            self.params.color_seed = fresh_seed();
        }
        self.preview_id = self.preview_id.wrapping_add(1);
        self.busy = true;
        let _ = self.preview_tx.send(PreviewRequest {
            id: self.preview_id,
            params: self.params.clone(),
            width,
            height,
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(done) = self.preview_rx.try_recv() {
            if done.id != self.preview_id {
                continue;
            }
            if done.width == 0 || done.height == 0 {
                self.busy = false;
                continue;
            }
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [done.width as usize, done.height as usize],
                &done.rgba,
            );
            self.texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR));
            self.segment_count = done.segments;
            self.truncated = done.truncated;
            self.preview_ms = done.elapsed_ms;
            self.busy = false;
        }
        while let Ok(done) = self.export_rx.try_recv() {
            self.exporting = false;
            self.export_status = match done.error {
                Some(err) => format!("Export failed: {err}"),
                None => format!(
                    "Saved {} ({:.0} ms)",
                    done.path.display(),
                    done.elapsed_ms
                ),
            };
        }

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Polygon fractal");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls(ui);
                });
            });

        let mut preview_size = (0, 0);
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            preview_size = (
                preview_extent(available.x),
                preview_extent(available.y),
            );
            if let Some(texture) = &self.texture {
                let image = texture.size_vec2();
                let scale = (available.x / image.x).min(available.y / image.y);
                let size = image * scale;
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new((texture.id(), size)));
                });
            } else if self.busy {
                ui.centered_and_justified(|ui| {
                    ui.label("Rendering…");
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Click Update preview");
                });
            }
        });
        self.panel_size = preview_size;

        if self.busy || self.exporting {
            ctx.request_repaint();
        }
    }
}

fn preview_extent(available: f32) -> u32 {
    let raw = (available as u32).saturating_sub(8).max(32);
    (raw / 8) * 8
}

fn fresh_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(1);
    nanos ^ nanos.rotate_left(17) ^ 0xA5A5_5A5A_1234_5678
}

fn preview_worker(rx: Receiver<PreviewRequest>, tx: Sender<PreviewDone>) {
    while let Ok(mut request) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            request = newer;
        }
        let started = Instant::now();
        let output = generate(&request.params, request.width, request.height);
        let Some(pixmap) = render_pixmap(&output, &request.params, request.width, request.height)
        else {
            let _ = tx.send(PreviewDone {
                id: request.id,
                width: 0,
                height: 0,
                rgba: Vec::new(),
                segments: output.segments.len(),
                truncated: output.truncated,
                elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
            });
            continue;
        };
        let _ = tx.send(PreviewDone {
            id: request.id,
            width: pixmap.width(),
            height: pixmap.height(),
            rgba: straight_rgba(&pixmap),
            segments: output.segments.len(),
            truncated: output.truncated,
            elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
        });
    }
}

fn export_worker(rx: Receiver<ExportRequest>, tx: Sender<ExportDone>) {
    while let Ok(request) = rx.recv() {
        let started = Instant::now();
        let outcome = (|| {
            let output = generate(&request.params, request.width, request.height);
            let pixmap = render_pixmap(&output, &request.params, request.width, request.height)
                .ok_or_else(|| "could not allocate the image".to_string())?;
            write_png(&request.path, &pixmap)
        })();
        let _ = tx.send(ExportDone {
            path: request.path,
            elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
            error: outcome.err(),
        });
    }
}

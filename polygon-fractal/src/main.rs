mod geom;
mod render;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;

use eframe::egui;
use geom::{
    arrange, equalize, generate, set_side_count, ArrangementParams, BgMode, ColourMode, FillMode,
    FractalParams, Motif,
};
use render::{
    composite_arrangement, render_pixmap, straight_rgba, write_png, write_svg, StampCache,
};

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum EditorMode {
    Shape,
    Arrangement,
}

#[derive(Clone, Copy)]
enum MotifAction {
    Edit,
    Clone,
    Delete,
}

enum PreviewJob {
    Shape(FractalParams),
    Arrangement {
        arrangement: ArrangementParams,
        motifs: Vec<Motif>,
    },
}

struct PreviewRequest {
    id: u64,
    job: PreviewJob,
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
    arranged: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportFormat {
    Png,
    Svg,
}

enum ExportJob {
    Shape(FractalParams),
    Arrangement {
        arrangement: ArrangementParams,
        motifs: Vec<Motif>,
    },
}

struct ExportRequest {
    job: ExportJob,
    format: ExportFormat,
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
    mode: EditorMode,
    motifs: Vec<Motif>,
    next_motif_id: u64,
    editing_motif: Option<u64>,
    arrangement: ArrangementParams,
    show_stamps: bool,
    export_width: u32,
    export_height: u32,
    export_format: ExportFormat,
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
            mode: EditorMode::Shape,
            motifs: Vec::new(),
            next_motif_id: 0,
            editing_motif: None,
            arrangement: ArrangementParams::default(),
            show_stamps: false,
            export_width: 4096,
            export_height: 4096,
            export_format: ExportFormat::Png,
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
        match self.mode {
            EditorMode::Shape => self.shape_controls(ui),
            EditorMode::Arrangement => self.arrangement_controls(ui),
        }
        self.preview_controls(ui);
        self.export_controls(ui);
    }

    fn mode_toggle(&mut self, ui: &mut egui::Ui) {
        let before = self.mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, EditorMode::Shape, "Shape");
            ui.selectable_value(&mut self.mode, EditorMode::Arrangement, "Arrangement");
        });
        if before != EditorMode::Arrangement
            && self.mode == EditorMode::Arrangement
            && self.motifs.is_empty()
        {
            self.capture_motif(false);
        }
    }

    fn shape_controls(&mut self, ui: &mut egui::Ui) {
        if ui.button("Randomize").clicked() {
            self.params.randomize(fresh_seed());
            self.queue_preview();
        }

        let mut sides = self.params.side_lengths.len() as u32;
        if ui
            .add(egui::Slider::new(&mut sides, 3..=16).text("Sides"))
            .changed()
        {
            set_side_count(&mut self.params.side_lengths, sides as usize);
        }

        for (index, length) in self.params.side_lengths.iter_mut().enumerate() {
            ui.add(egui::Slider::new(length, 0.05..=4.0).text(format!("Side {}", index + 1)));
        }
        if ui.button("Equalize").clicked() {
            equalize(&mut self.params.side_lengths);
        }

        ui.add(egui::Slider::new(&mut self.params.shrink, 0.05..=0.95).text("Shrink"));
        ui.add(egui::Slider::new(&mut self.params.depth, 1..=12).text("Depth"));
        shrink_by_depth_controls(
            ui,
            self.params.shrink,
            self.params.depth,
            &mut self.params.shrink_by_depth,
        );
        ui.add(egui::Slider::new(&mut self.params.stroke_fraction, 0.002..=0.08).text("Stroke"));
        ui.add(egui::Slider::new(&mut self.params.stroke_taper, 0.0..=1.0).text("Taper"));
        ui.add(egui::Slider::new(&mut self.params.corner_radius, 0.0..=1.0).text("Corner"));
        ui.add(egui::Slider::new(&mut self.params.jitter, 0.0..=0.5).text("Jitter"))
            .on_hover_text("High jitter opens instance outlines slightly.");
        ui.horizontal(|ui| {
            ui.label("Jitter seed");
            ui.add(egui::DragValue::new(&mut self.params.jitter_seed).speed(1.0));
        });
        ui.add(egui::Slider::new(&mut self.params.stroke_opacity, 0.0..=1.0).text("Opacity"));
        ui.add(egui::Slider::new(&mut self.params.opacity_fade, 0.0..=1.0).text("Opacity fade"));
        ui.label("Fill");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.fill_mode, FillMode::None, "None");
            ui.selectable_value(&mut self.params.fill_mode, FillMode::Solid, "Solid");
            ui.selectable_value(&mut self.params.fill_mode, FillMode::Palette, "Palette");
        });
        ui.add(egui::Slider::new(&mut self.params.fill_opacity, 0.0..=1.0).text("Fill opacity"));
        ui.add(egui::Slider::new(&mut self.params.rotation_deg, -180.0..=180.0).text("Rotation"));
        ui.add(egui::Slider::new(&mut self.params.twist_deg, -180.0..=180.0).text("Twist"));
        ui.add(egui::Slider::new(&mut self.params.symmetry, 1..=24).text("Symmetry"));
        ui.checkbox(&mut self.params.mirror, "Mirror");
        ui.add(egui::Slider::new(&mut self.params.glow, 0.0..=2.0).text("Glow"));
        ui.add(egui::Slider::new(&mut self.params.glow_radius, 0.0..=64.0).text("Glow radius"));

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.inward, false, "Outward");
            ui.selectable_value(&mut self.params.inward, true, "Inward");
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Single, "Single");
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Gradient, "Gradient");
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Random, "Random");
            ui.selectable_value(&mut self.params.colour_mode, ColourMode::Palette, "Palette");
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
                    ui.checkbox(&mut self.params.gradient_hsv, "HSV");
                });
            }
            ColourMode::Random => {
                ui.label("Each line gets its own colour.");
            }
            ColourMode::Palette => {
                ui.checkbox(&mut self.params.gradient_hsv, "HSV");
                let stop_count = self.params.palette.len();
                let mut remove_at = None;
                for index in 0..stop_count {
                    ui.horizontal(|ui| {
                        ui.label(format!("Stop {}", index + 1));
                        ui.color_edit_button_srgb(&mut self.params.palette[index]);
                        if ui
                            .add_enabled(stop_count > 2, egui::Button::new("Remove"))
                            .clicked()
                        {
                            remove_at = Some(index);
                        }
                    });
                }
                if let Some(index) = remove_at {
                    self.params.palette.remove(index);
                }
                if ui.button("Add stop").clicked() {
                    let next = self
                        .params
                        .palette
                        .last()
                        .copied()
                        .unwrap_or(self.params.color_end);
                    self.params.palette.push(next);
                }
            }
        }
        background_controls(
            ui,
            &mut self.params.background,
            &mut self.params.bg_mode,
            &mut self.params.bg_end,
            &mut self.params.bg_angle,
        );

        ui.separator();
        ui.label("Motifs");
        if let Some(id) = self.editing_motif {
            let editing_name = self
                .motifs
                .iter()
                .find(|motif| motif.id == id)
                .map(|motif| motif.name.clone());
            if let Some(name) = editing_name {
                ui.label(format!("Editing {name}"));
                if ui.button("Update motif").clicked() {
                    self.update_motif();
                }
            }
        }
        if ui.button("Save motif").clicked() {
            self.capture_motif(true);
        }
        let motif_ids: Vec<u64> = self.motifs.iter().map(|motif| motif.id).collect();
        for id in motif_ids {
            let Some(name) = self
                .motifs
                .iter()
                .find(|motif| motif.id == id)
                .map(|motif| motif.name.clone())
            else {
                continue;
            };
            let mut action = None;
            ui.push_id(id, |ui| {
                ui.label(&name);
                ui.horizontal(|ui| {
                    if ui.button("Edit").clicked() {
                        action = Some(MotifAction::Edit);
                    }
                    if ui.button("Clone").clicked() {
                        action = Some(MotifAction::Clone);
                    }
                    if ui
                        .add_enabled(self.motifs.len() > 1, egui::Button::new("Delete"))
                        .clicked()
                    {
                        action = Some(MotifAction::Delete);
                    }
                });
            });
            match action {
                Some(MotifAction::Edit) => self.edit_motif(id),
                Some(MotifAction::Clone) => self.clone_motif(id),
                Some(MotifAction::Delete) => self.delete_motif(id),
                None => {}
            }
        }
    }

    fn arrangement_controls(&mut self, ui: &mut egui::Ui) {
        if ui.button("Randomize").clicked() {
            self.arrangement.randomize(fresh_seed());
            self.queue_preview();
        }

        let mut sides = self.arrangement.side_lengths.len() as u32;
        if ui
            .add(egui::Slider::new(&mut sides, 3..=16).text("Sides"))
            .changed()
        {
            set_side_count(&mut self.arrangement.side_lengths, sides as usize);
        }
        for (index, length) in self.arrangement.side_lengths.iter_mut().enumerate() {
            ui.add(egui::Slider::new(length, 0.05..=4.0).text(format!("Side {}", index + 1)));
        }
        if ui.button("Equalize").clicked() {
            equalize(&mut self.arrangement.side_lengths);
        }
        ui.add(egui::Slider::new(&mut self.arrangement.shrink, 0.05..=0.95).text("Shrink"));
        ui.add(
            egui::Slider::new(&mut self.arrangement.gap, -geom::GAP_LIMIT..=geom::GAP_LIMIT)
                .text("Gap"),
        );
        ui.add(egui::Slider::new(&mut self.arrangement.depth, 0..=12).text("Depth"));
        shrink_by_depth_controls(
            ui,
            self.arrangement.shrink,
            self.arrangement.depth,
            &mut self.arrangement.shrink_by_depth,
        );
        ui.add(
            egui::Slider::new(&mut self.arrangement.rotation_deg, -180.0..=180.0).text("Rotation"),
        );
        ui.add(egui::Slider::new(&mut self.arrangement.twist_deg, -180.0..=180.0).text("Twist"));
        ui.add(egui::Slider::new(&mut self.arrangement.symmetry, 1..=24).text("Symmetry"));
        ui.checkbox(&mut self.arrangement.mirror, "Mirror");
        ui.add(egui::Slider::new(&mut self.arrangement.glow, 0.0..=2.0).text("Glow"));
        ui.add(egui::Slider::new(&mut self.arrangement.glow_radius, 0.0..=64.0).text("Glow radius"));
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.arrangement.inward, false, "Outward");
            ui.selectable_value(&mut self.arrangement.inward, true, "Inward");
        });
        background_controls(
            ui,
            &mut self.arrangement.background,
            &mut self.arrangement.bg_mode,
            &mut self.arrangement.bg_end,
            &mut self.arrangement.bg_angle,
        );
        self.sync_depth_motifs();
        ui.separator();
        ui.label("Motif at each depth");
        let depth = self.arrangement.depth;
        let choices: Vec<(u64, String)> = self
            .motifs
            .iter()
            .map(|motif| (motif.id, motif.name.clone()))
            .collect();
        for level in 0..=depth {
            let slot = &mut self.arrangement.motif_by_depth[level as usize];
            let selected = choices
                .iter()
                .find(|(id, _)| id == slot)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| "Missing".to_string());
            egui::ComboBox::from_id_salt(("motif-depth", level))
                .selected_text(format!("Depth {level}: {selected}"))
                .show_ui(ui, |ui| {
                    for (id, name) in &choices {
                        ui.selectable_value(slot, *id, name);
                    }
                });
        }
    }

    fn preview_controls(&mut self, ui: &mut egui::Ui) {
        if ui
            .add_enabled(!self.busy, egui::Button::new("Update preview"))
            .on_hover_text("Draw the preview with the current settings")
            .clicked()
        {
            self.queue_preview();
        }

        let lengths = match self.mode {
            EditorMode::Shape => &self.params.side_lengths,
            EditorMode::Arrangement => &self.arrangement.side_lengths,
        };
        let gap = geom::close_seed(lengths).gap_ratio;
        ui.separator();
        ui.label(format!("Closure gap: {gap:.2}% of perimeter", gap = gap * 100.0));
        if self.show_stamps {
            ui.label(format!("Stamps: {}", self.segment_count));
        } else {
            ui.label(format!("Segments: {}", self.segment_count));
        }
        ui.label(format!("Preview: {:.0} ms", self.preview_ms));
        if self.truncated {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Limit reached. Lower the depth or shrink for a complete figure.",
            );
        }
        if self.busy {
            ui.label("Rendering preview…");
        }
    }

    fn export_controls(&mut self, ui: &mut egui::Ui) {
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
        if self.mode != EditorMode::Shape {
            self.export_format = ExportFormat::Png;
        }
        ui.label("Format");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.export_format, ExportFormat::Png, "PNG");
            ui.add_enabled_ui(self.mode == EditorMode::Shape, |ui| {
                ui.selectable_value(&mut self.export_format, ExportFormat::Svg, "SVG");
            })
            .response
            .on_disabled_hover_text("SVG export is only available in Shape mode");
        });
        let svg = self.export_format == ExportFormat::Svg;
        let export = ui
            .add_enabled(
                !self.exporting,
                egui::Button::new(if svg { "Export SVG" } else { "Export PNG" }),
            )
            .on_hover_text(if svg {
                "Choose a path and write a vector SVG"
            } else {
                "Choose a path and write an anti-aliased PNG"
            });
        if export.clicked() {
            let dialog = rfd::FileDialog::new().set_title("Export fractal");
            let dialog = if svg {
                dialog
                    .set_file_name("fractal.svg")
                    .add_filter("SVG image", &["svg"])
            } else {
                dialog
                    .set_file_name("fractal.png")
                    .add_filter("PNG image", &["png"])
            };
            if let Some(path) = dialog.save_file() {
                self.exporting = true;
                self.export_status = "Exporting…".to_string();
                let job = match self.mode {
                    EditorMode::Shape => ExportJob::Shape(self.params.clone()),
                    EditorMode::Arrangement => {
                        self.sync_depth_motifs();
                        ExportJob::Arrangement {
                            arrangement: self.arrangement.clone(),
                            motifs: self.motifs.clone(),
                        }
                    }
                };
                let _ = self.export_tx.send(ExportRequest {
                    job,
                    format: self.export_format,
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

    fn capture_motif(&mut self, edit: bool) {
        self.next_motif_id = self.next_motif_id.saturating_add(1);
        let id = self.next_motif_id;
        self.motifs.push(Motif {
            id,
            name: format!("Motif {id}"),
            revision: 1,
            params: self.params.clone(),
        });
        if edit {
            self.editing_motif = Some(id);
        }
    }

    fn update_motif(&mut self) {
        let Some(id) = self.editing_motif else {
            return;
        };
        let params = self.params.clone();
        if let Some(motif) = self.motifs.iter_mut().find(|motif| motif.id == id) {
            motif.params = params;
            motif.revision = motif.revision.wrapping_add(1);
        }
    }

    fn edit_motif(&mut self, id: u64) {
        let Some(motif) = self.motifs.iter().find(|motif| motif.id == id) else {
            return;
        };
        self.params = motif.params.clone();
        self.editing_motif = Some(id);
        self.mode = EditorMode::Shape;
    }

    fn clone_motif(&mut self, id: u64) {
        let Some(source) = self.motifs.iter().find(|motif| motif.id == id).cloned() else {
            return;
        };
        self.next_motif_id = self.next_motif_id.saturating_add(1);
        let new_id = self.next_motif_id;
        let motif = Motif {
            id: new_id,
            name: format!("{} copy", source.name),
            revision: 1,
            params: source.params,
        };
        self.params = motif.params.clone();
        self.editing_motif = Some(new_id);
        self.motifs.push(motif);
        self.mode = EditorMode::Shape;
    }

    fn delete_motif(&mut self, id: u64) {
        if self.motifs.len() <= 1 {
            return;
        }
        self.motifs.retain(|motif| motif.id != id);
        if self.editing_motif == Some(id) {
            self.editing_motif = None;
        }
        let Some(fallback) = self.motifs.first().map(|motif| motif.id) else {
            return;
        };
        for slot in &mut self.arrangement.motif_by_depth {
            if *slot == id || !self.motifs.iter().any(|motif| motif.id == *slot) {
                *slot = fallback;
            }
        }
    }

    fn sync_depth_motifs(&mut self) {
        let fallback = self.motifs.first().map(|motif| motif.id).unwrap_or(1);
        if self.arrangement.motif_by_depth.is_empty() {
            self.arrangement.motif_by_depth.push(fallback);
        }
        let slots = self.arrangement.depth as usize + 1;
        if self.arrangement.motif_by_depth.len() < slots {
            let fill = *self.arrangement.motif_by_depth.last().unwrap_or(&fallback);
            self.arrangement.motif_by_depth.resize(slots, fill);
        }
    }

    fn queue_preview(&mut self) {
        let (width, height) = self.panel_size;
        if width < 32 || height < 32 {
            return;
        }
        if self.mode == EditorMode::Shape && self.params.colour_mode == ColourMode::Random {
            self.params.color_seed = fresh_seed();
        }
        self.preview_id = self.preview_id.wrapping_add(1);
        self.busy = true;
        let job = match self.mode {
            EditorMode::Shape => PreviewJob::Shape(self.params.clone()),
            EditorMode::Arrangement => {
                self.sync_depth_motifs();
                PreviewJob::Arrangement {
                    arrangement: self.arrangement.clone(),
                    motifs: self.motifs.clone(),
                }
            }
        };
        let _ = self.preview_tx.send(PreviewRequest {
            id: self.preview_id,
            job,
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
            self.show_stamps = done.arranged;
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
                self.mode_toggle(ui);
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

fn shrink_by_depth_controls(ui: &mut egui::Ui, base: f32, depth: u32, by_depth: &mut Vec<f32>) {
    let mut enabled = !by_depth.is_empty();
    if ui.checkbox(&mut enabled, "Per-depth shrink").changed() {
        if enabled {
            *by_depth = vec![base; depth as usize + 1];
        } else {
            by_depth.clear();
        }
    }
    if by_depth.is_empty() {
        return;
    }
    let slots = depth as usize + 1;
    if by_depth.len() != slots {
        by_depth.resize(slots, base);
    }
    for (level, value) in by_depth.iter_mut().enumerate() {
        ui.add(egui::Slider::new(value, 0.05..=0.95).text(format!("Depth {level} shrink")));
    }
}

fn background_controls(
    ui: &mut egui::Ui,
    background: &mut [u8; 3],
    mode: &mut BgMode,
    end: &mut [u8; 3],
    angle: &mut f32,
) {
    ui.horizontal(|ui| {
        ui.label("Background");
        ui.color_edit_button_srgb(background);
    });
    ui.horizontal(|ui| {
        ui.selectable_value(mode, BgMode::Solid, "Solid");
        ui.selectable_value(mode, BgMode::Linear, "Linear");
        ui.selectable_value(mode, BgMode::Radial, "Radial");
    });
    if *mode != BgMode::Solid {
        ui.horizontal(|ui| {
            ui.label("End");
            ui.color_edit_button_srgb(end);
        });
        if *mode == BgMode::Linear {
            ui.add(egui::Slider::new(angle, -180.0..=180.0).text("Angle"));
        }
    }
}

fn fresh_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(1);
    nanos ^ nanos.rotate_left(17) ^ 0xA5A5_5A5A_1234_5678
}

fn preview_worker(rx: Receiver<PreviewRequest>, tx: Sender<PreviewDone>) {
    let mut stamps = StampCache::default();
    while let Ok(mut request) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            request = newer;
        }
        let started = Instant::now();
        match request.job {
            PreviewJob::Shape(params) => {
                let output = generate(&params, request.width, request.height);
                let Some(pixmap) = render_pixmap(&output, &params, request.width, request.height)
                else {
                    let _ = tx.send(PreviewDone {
                        id: request.id,
                        width: 0,
                        height: 0,
                        rgba: Vec::new(),
                        segments: output.segments.len(),
                        truncated: output.truncated,
                        elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                        arranged: false,
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
                    arranged: false,
                });
            }
            PreviewJob::Arrangement {
                arrangement,
                motifs,
            } => {
                let output = arrange(&arrangement, request.width, request.height);
                let Some(pixmap) = composite_arrangement(
                    &mut stamps,
                    &motifs,
                    &output,
                    arrangement.background,
                    arrangement.bg_mode,
                    arrangement.bg_end,
                    arrangement.bg_angle,
                    request.width,
                    request.height,
                    1024,
                    arrangement.glow,
                    arrangement.glow_radius,
                ) else {
                    let _ = tx.send(PreviewDone {
                        id: request.id,
                        width: 0,
                        height: 0,
                        rgba: Vec::new(),
                        segments: output.placements.len(),
                        truncated: output.truncated,
                        elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                        arranged: true,
                    });
                    continue;
                };
                let _ = tx.send(PreviewDone {
                    id: request.id,
                    width: pixmap.width(),
                    height: pixmap.height(),
                    rgba: straight_rgba(&pixmap),
                    segments: output.placements.len(),
                    truncated: output.truncated,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    arranged: true,
                });
            }
        }
    }
}

fn export_worker(rx: Receiver<ExportRequest>, tx: Sender<ExportDone>) {
    let mut stamps = StampCache::default();
    while let Ok(request) = rx.recv() {
        let started = Instant::now();
        let outcome = (|| match request.job {
            ExportJob::Shape(params) => match request.format {
                ExportFormat::Png => {
                    let output = generate(&params, request.width, request.height);
                    let pixmap = render_pixmap(&output, &params, request.width, request.height)
                        .ok_or_else(|| "could not allocate the image".to_string())?;
                    write_png(&request.path, &pixmap)
                }
                ExportFormat::Svg => {
                    let output = generate(&params, request.width, request.height);
                    write_svg(
                        &request.path,
                        &output,
                        &params,
                        request.width,
                        request.height,
                    )
                }
            },
            ExportJob::Arrangement {
                arrangement,
                motifs,
            } => {
                if request.format == ExportFormat::Svg {
                    return Err("SVG export is only available in Shape mode".to_string());
                }
                let output = arrange(&arrangement, request.width, request.height);
                let pixmap = composite_arrangement(
                    &mut stamps,
                    &motifs,
                    &output,
                    arrangement.background,
                    arrangement.bg_mode,
                    arrangement.bg_end,
                    arrangement.bg_angle,
                    request.width,
                    request.height,
                    4096,
                    arrangement.glow,
                    arrangement.glow_radius,
                )
                .ok_or_else(|| "could not allocate the image".to_string())?;
                write_png(&request.path, &pixmap)
            }
        })();
        let _ = tx.send(ExportDone {
            path: request.path,
            elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
            error: outcome.err(),
        });
    }
}

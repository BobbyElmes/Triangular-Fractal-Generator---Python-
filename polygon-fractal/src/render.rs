//! Anti-aliased outline rendering and PNG/SVG export.
//!
//! Preview and export share one path: segments are binned into tiles and
//! stroked in parallel, then copied into a single pixmap.

use crate::geom::{
    fill_color, ArrangementOutput, BgMode, Bounds, FillMode, FractalParams, InstanceOutline, Motif,
    Output, Segment, StampPlacement,
};
use png::{BitDepth, ColorType, Encoder};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use tiny_skia::{
    BlendMode, Color, FillRule, FilterQuality, GradientStop, LineCap, LineJoin, LinearGradient,
    Paint, PathBuilder, Pixmap, PixmapPaint, Point, RadialGradient, Rect, Shader, SpreadMode,
    Stroke, Transform,
};

const TILE: u32 = 512;
const MARGIN: f32 = 0.04;
const PATH_CHUNK: usize = 8_000;
const MAX_GLOW_RADIUS: f32 = 64.0;

struct View {
    scale: f32,
    tx: f32,
    ty: f32,
    height: f32,
}

struct PxSeg {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    depth: u16,
    color: [u8; 3],
}

pub fn render_pixmap(output: &Output, params: &FractalParams, width: u32, height: u32) -> Option<Pixmap> {
    render_pixmap_inner(
        output,
        width,
        height,
        Some(params.background),
        params.bg_mode,
        params.bg_end,
        params.bg_angle,
        MARGIN,
        params.stroke_opacity,
        params.opacity_fade,
        params.stroke_taper,
        params.depth,
        params.glow,
        params.glow_radius,
        params,
    )
}

fn render_pixmap_inner(
    output: &Output,
    width: u32,
    height: u32,
    background: Option<[u8; 3]>,
    bg_mode: BgMode,
    bg_end: [u8; 3],
    bg_angle: f32,
    margin: f32,
    stroke_opacity: f32,
    opacity_fade: f32,
    stroke_taper: f32,
    max_depth: u32,
    glow: f32,
    glow_radius: f32,
    fill_params: &FractalParams,
) -> Option<Pixmap> {
    if width == 0 || height == 0 {
        return None;
    }
    let mut full = Pixmap::new(width, height)?;
    let clear = clear_color(background);
    fill_background(&mut full, background, bg_mode, bg_end, bg_angle);

    let view = fit_view(width, height, output.bounds, output.stroke_world, margin);
    let stroke_px = (output.stroke_world * view.scale).max(0.25);
    let px_segs = pixel_segments(&output.segments, &view);
    // Fills land on the canvas before tiles are stroked, and only when a fill was requested.
    let draw_fills = fill_params.fill_mode != FillMode::None
        && !output.outlines.is_empty()
        && fill_alpha(fill_params.fill_opacity) > 0;
    if draw_fills {
        fill_outlines(&mut full, &output.outlines, &view, fill_params);
    }

    let tiles_x = (width + TILE - 1) / TILE;
    let tiles_y = (height + TILE - 1) / TILE;
    let tile_count = (tiles_x * tiles_y) as usize;
    let mut bins: Vec<Vec<PxSeg>> = (0..tile_count).map(|_| Vec::new()).collect();
    let pad = stroke_px * 0.5 + 1.0;
    for segment in px_segs {
        let min_x = (segment.x0.min(segment.x1) - pad).floor().max(0.0) as u32;
        let max_x = (segment.x0.max(segment.x1) + pad).ceil().max(0.0) as u32;
        let min_y = (segment.y0.min(segment.y1) - pad).floor().max(0.0) as u32;
        let max_y = (segment.y0.max(segment.y1) + pad).ceil().max(0.0) as u32;
        if min_x >= width || min_y >= height {
            continue;
        }
        let x0 = min_x.min(width - 1) / TILE;
        let x1 = max_x.min(width - 1) / TILE;
        let y0 = min_y.min(height - 1) / TILE;
        let y1 = max_y.min(height - 1) / TILE;
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                bins[(ty * tiles_x + tx) as usize].push(PxSeg {
                    x0: segment.x0,
                    y0: segment.y0,
                    x1: segment.x1,
                    y1: segment.y1,
                    depth: segment.depth,
                    color: segment.color,
                });
            }
        }
    }

    // Blit in small batches so a 16k export does not keep every tile alive at once.
    const BATCH: usize = 8;
    for batch_start in (0..tile_count).step_by(BATCH) {
        let batch_end = (batch_start + BATCH).min(tile_count);
        let rendered: Vec<(u32, u32, Pixmap)> = (batch_start..batch_end)
            .into_par_iter()
            .filter_map(|index| {
                let tx = index as u32 % tiles_x;
                let ty = index as u32 / tiles_x;
                let origin_x = tx * TILE;
                let origin_y = ty * TILE;
                let tile_w = (width - origin_x).min(TILE);
                let tile_h = (height - origin_y).min(TILE);
                let mut tile = Pixmap::new(tile_w, tile_h)?;
                if draw_fills {
                    copy_tile_from(&mut tile, &full, origin_x, origin_y);
                } else {
                    tile.fill(clear);
                    // Tiles are copied over the canvas, so a non-solid fill has to be painted here too.
                    if bg_mode != BgMode::Solid {
                        if let Some(start) = background {
                            paint_background_gradient(
                                &mut tile,
                                width as f32,
                                height as f32,
                                start,
                                bg_mode,
                                bg_end,
                                bg_angle,
                                origin_x as f32,
                                origin_y as f32,
                            );
                        }
                    }
                }
                stroke_tile(
                    &mut tile,
                    &bins[index],
                    origin_x as f32,
                    origin_y as f32,
                    stroke_px,
                    stroke_opacity,
                    opacity_fade,
                    stroke_taper,
                    max_depth,
                );
                Some((origin_x, origin_y, tile))
            })
            .collect();
        for (origin_x, origin_y, tile) in rendered {
            blit(&mut full, origin_x, origin_y, &tile);
        }
    }

    apply_glow(&mut full, glow, glow_radius);
    Some(full)
}

pub fn write_png(path: &Path, pixmap: &Pixmap) -> Result<(), String> {
    let file = File::create(path).map_err(|err| err.to_string())?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, pixmap.width(), pixmap.height());
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|err| err.to_string())?;
    writer
        .write_image_data(&straight_rgba(pixmap))
        .map_err(|err| err.to_string())
}

/// Write a Shape-mode figure as SVG in the same pixel frame as [`render_pixmap`].
///
/// Each segment keeps its colour. Stroke width starts from `stroke_world * view.scale`
/// (with the same minimum as the raster path) and follows per-depth taper and opacity.
pub fn write_svg(
    path: &Path,
    output: &Output,
    params: &FractalParams,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("could not write an empty svg".to_string());
    }
    let view = fit_view(width, height, output.bounds, output.stroke_world, MARGIN);
    let stroke_px = (output.stroke_world * view.scale).max(0.25);
    let mut px_segs = pixel_segments(&output.segments, &view);
    px_segs.sort_by_key(|segment| segment.depth);

    let file = File::create(path).map_err(|err| err.to_string())?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#).map_err(|err| err.to_string())?;
    writeln!(
        writer,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"##
    )
    .map_err(|err| err.to_string())?;
    write_svg_background(&mut writer, params, width, height).map_err(|err| err.to_string())?;

    let mut start = 0;
    while start < px_segs.len() {
        let depth = px_segs[start].depth;
        let color = px_segs[start].color;
        let mut end = start + 1;
        while end < px_segs.len() && px_segs[end].depth == depth && px_segs[end].color == color {
            end += 1;
        }
        let stroke_width = tapered_stroke_width(stroke_px, params.stroke_taper, depth, params.depth);
        let alpha = stroke_alpha(params.stroke_opacity, params.opacity_fade, depth, params.depth);
        write_svg_path(&mut writer, &px_segs[start..end], color, alpha, stroke_width)
            .map_err(|err| err.to_string())?;
        start = end;
    }

    writeln!(writer, "</svg>").map_err(|err| err.to_string())?;
    writer.flush().map_err(|err| err.to_string())
}

fn write_svg_path(
    out: &mut impl Write,
    segments: &[PxSeg],
    color: [u8; 3],
    alpha: u8,
    stroke_width: f32,
) -> std::io::Result<()> {
    let stroke_width = if stroke_width.is_finite() {
        stroke_width
    } else {
        0.25
    };
    let opacity = alpha as f32 / 255.0;
    let mut started = false;
    for segment in segments {
        if !segment.x0.is_finite()
            || !segment.y0.is_finite()
            || !segment.x1.is_finite()
            || !segment.y1.is_finite()
        {
            continue;
        }
        if !started {
            write!(
                out,
                "<path fill=\"none\" stroke=\"#{:02x}{:02x}{:02x}\" stroke-opacity=\"{opacity:.6}\" stroke-width=\"{stroke_width:.4}\" stroke-linecap=\"round\" d=\"",
                color[0], color[1], color[2]
            )?;
            started = true;
        }
        write!(
            out,
            "M {:.3} {:.3} L {:.3} {:.3} ",
            segment.x0, segment.y0, segment.x1, segment.y1
        )?;
    }
    if started {
        writeln!(out, "\"/>")?;
    }
    Ok(())
}

/// Premultiplied tiny-skia pixels become straight alpha for PNG and egui.
pub fn straight_rgba(pixmap: &Pixmap) -> Vec<u8> {
    let src = pixmap.data();
    let mut out = vec![0_u8; src.len()];
    for i in (0..src.len()).step_by(4) {
        let alpha = src[i + 3] as u16;
        if alpha == 0 {
            continue;
        } else if alpha == 255 {
            out[i..i + 4].copy_from_slice(&src[i..i + 4]);
        } else {
            out[i] = ((src[i] as u16 * 255) / alpha) as u8;
            out[i + 1] = ((src[i + 1] as u16 * 255) / alpha) as u8;
            out[i + 2] = ((src[i + 2] as u16 * 255) / alpha) as u8;
            out[i + 3] = src[i + 3];
        }
    }
    out
}

fn clear_color(background: Option<[u8; 3]>) -> Color {
    match background {
        Some(bg) => Color::from_rgba8(bg[0], bg[1], bg[2], 255),
        None => Color::from_rgba8(0, 0, 0, 0),
    }
}

/// Solid backgrounds use [`Pixmap::fill`]. Linear and radial modes paint a shader over that fill.
fn fill_background(
    pixmap: &mut Pixmap,
    background: Option<[u8; 3]>,
    mode: BgMode,
    end: [u8; 3],
    angle: f32,
) {
    pixmap.fill(clear_color(background));
    if let Some(start) = background {
        if mode != BgMode::Solid {
            paint_background_gradient(
                pixmap,
                pixmap.width() as f32,
                pixmap.height() as f32,
                start,
                mode,
                end,
                angle,
                0.0,
                0.0,
            );
        }
    }
}

fn paint_background_gradient(
    pixmap: &mut Pixmap,
    canvas_w: f32,
    canvas_h: f32,
    start: [u8; 3],
    mode: BgMode,
    end: [u8; 3],
    angle: f32,
    origin_x: f32,
    origin_y: f32,
) -> bool {
    let Some(mut shader) = background_shader(canvas_w, canvas_h, start, end, mode, angle) else {
        return false;
    };
    if origin_x != 0.0 || origin_y != 0.0 {
        shader.transform(Transform::from_translate(-origin_x, -origin_y));
    }
    let Some(rect) = Rect::from_xywh(0.0, 0.0, pixmap.width() as f32, pixmap.height() as f32) else {
        return false;
    };
    let mut paint = Paint::default();
    paint.shader = shader;
    paint.anti_alias = false;
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    true
}

fn background_shader(
    width: f32,
    height: f32,
    start: [u8; 3],
    end: [u8; 3],
    mode: BgMode,
    angle: f32,
) -> Option<Shader<'static>> {
    let stops = vec![
        GradientStop::new(0.0, Color::from_rgba8(start[0], start[1], start[2], 255)),
        GradientStop::new(1.0, Color::from_rgba8(end[0], end[1], end[2], 255)),
    ];
    match mode {
        BgMode::Solid => None,
        BgMode::Linear => {
            let (x0, y0, x1, y1) = linear_endpoints(width, height, angle);
            LinearGradient::new(
                Point::from_xy(x0, y0),
                Point::from_xy(x1, y1),
                stops,
                SpreadMode::Pad,
                Transform::identity(),
            )
        }
        BgMode::Radial => {
            let center = Point::from_xy(width * 0.5, height * 0.5);
            RadialGradient::new(
                center,
                center,
                radial_radius(width, height),
                stops,
                SpreadMode::Pad,
                Transform::identity(),
            )
        }
    }
}

/// 0 degrees runs left to right. The segment spans the canvas so the stops meet its edges.
fn linear_endpoints(width: f32, height: f32, angle_deg: f32) -> (f32, f32, f32, f32) {
    let angle = if angle_deg.is_finite() { angle_deg } else { 0.0 };
    let (sin, cos) = angle.to_radians().sin_cos();
    let cx = width * 0.5;
    let cy = height * 0.5;
    let half = (width * 0.5 * cos.abs() + height * 0.5 * sin.abs()).max(0.5);
    (
        cx - cos * half,
        cy - sin * half,
        cx + cos * half,
        cy + sin * half,
    )
}

fn radial_radius(width: f32, height: f32) -> f32 {
    (width * 0.5).hypot(height * 0.5).max(1.0)
}

fn write_svg_background(
    writer: &mut impl Write,
    params: &FractalParams,
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    let start = hex_colour(params.background);
    match params.bg_mode {
        BgMode::Solid => {
            writeln!(
                writer,
                r#"<rect width="{width}" height="{height}" fill="{start}"/>"#
            )
        }
        BgMode::Linear => {
            let (x1, y1, x2, y2) = linear_endpoints(width as f32, height as f32, params.bg_angle);
            let end = hex_colour(params.bg_end);
            writeln!(writer, "<defs>")?;
            writeln!(
                writer,
                r#"<linearGradient id="bg" gradientUnits="userSpaceOnUse" x1="{x1:.3}" y1="{y1:.3}" x2="{x2:.3}" y2="{y2:.3}">"#
            )?;
            writeln!(writer, r#"<stop offset="0" stop-color="{start}"/>"#)?;
            writeln!(writer, r#"<stop offset="1" stop-color="{end}"/>"#)?;
            writeln!(writer, "</linearGradient>")?;
            writeln!(writer, "</defs>")?;
            writeln!(
                writer,
                r#"<rect width="{width}" height="{height}" fill="url(#bg)"/>"#
            )
        }
        BgMode::Radial => {
            let cx = width as f32 * 0.5;
            let cy = height as f32 * 0.5;
            let radius = radial_radius(width as f32, height as f32);
            let end = hex_colour(params.bg_end);
            writeln!(writer, "<defs>")?;
            writeln!(
                writer,
                r#"<radialGradient id="bg" gradientUnits="userSpaceOnUse" cx="{cx:.3}" cy="{cy:.3}" r="{radius:.3}">"#
            )?;
            writeln!(writer, r#"<stop offset="0" stop-color="{start}"/>"#)?;
            writeln!(writer, r#"<stop offset="1" stop-color="{end}"/>"#)?;
            writeln!(writer, "</radialGradient>")?;
            writeln!(writer, "</defs>")?;
            writeln!(
                writer,
                r#"<rect width="{width}" height="{height}" fill="url(#bg)"/>"#
            )
        }
    }
}

fn hex_colour(color: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2])
}

fn fit_view(width: u32, height: u32, bounds: Bounds, stroke_world: f32, margin: f32) -> View {
    let pad = stroke_world * 0.5;
    let min_x = bounds.min_x - pad;
    let min_y = bounds.min_y - pad;
    let box_w = (bounds.max_x - bounds.min_x + stroke_world).max(1e-4);
    let box_h = (bounds.max_y - bounds.min_y + stroke_world).max(1e-4);
    let margin = margin.clamp(0.0, 0.45);
    let avail_w = width as f32 * (1.0 - 2.0 * margin);
    let avail_h = height as f32 * (1.0 - 2.0 * margin);
    let scale = (avail_w / box_w).min(avail_h / box_h);
    let used_w = box_w * scale;
    let used_h = box_h * scale;
    View {
        scale,
        tx: (width as f32 - used_w) * 0.5 - min_x * scale,
        ty: (height as f32 - used_h) * 0.5 - min_y * scale,
        height: height as f32,
    }
}

fn pixel_segments(segments: &[Segment], view: &View) -> Vec<PxSeg> {
    segments
        .iter()
        .map(|segment| {
            let (x0, y0) = world_to_px(view, segment.x0, segment.y0);
            let (x1, y1) = world_to_px(view, segment.x1, segment.y1);
            PxSeg {
                x0,
                y0,
                x1,
                y1,
                depth: segment.depth,
                color: segment.color,
            }
        })
        .collect()
}

fn fill_alpha(opacity: f32) -> u8 {
    let opacity = if opacity.is_finite() {
        opacity.clamp(0.0, 1.0)
    } else {
        1.0
    };
    (opacity * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Back-to-front closed fills in the same pixel frame as the strokes.
fn fill_outlines(
    pixmap: &mut Pixmap,
    outlines: &[InstanceOutline],
    view: &View,
    params: &FractalParams,
) {
    let alpha = fill_alpha(params.fill_opacity);
    if alpha == 0 {
        return;
    }
    let mut order: Vec<usize> = (0..outlines.len()).collect();
    order.sort_by_key(|&index| outlines[index].depth);
    for index in order {
        let outline = &outlines[index];
        if outline.verts.len() < 3 {
            continue;
        }
        if outline
            .verts
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            continue;
        }
        let mut builder = PathBuilder::new();
        let (x, y) = world_to_px(view, outline.verts[0].x, outline.verts[0].y);
        builder.move_to(x, y);
        for point in &outline.verts[1..] {
            let (x, y) = world_to_px(view, point.x, point.y);
            builder.line_to(x, y);
        }
        builder.close();
        let Some(path) = builder.finish() else {
            continue;
        };
        let color = fill_color(params, outline.depth as u32);
        let mut paint = Paint::default();
        paint.set_color_rgba8(color[0], color[1], color[2], alpha);
        paint.anti_alias = true;
        let _ = pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }
}

fn world_to_px(view: &View, x: f32, y: f32) -> (f32, f32) {
    let px = x * view.scale + view.tx;
    let py = view.height - (y * view.scale + view.ty);
    (px, py)
}

fn tapered_stroke_width(stroke_px: f32, stroke_taper: f32, depth: u16, max_depth: u32) -> f32 {
    let max_depth = (max_depth as f32).max(1.0);
    let width = stroke_px * (1.0 - stroke_taper * depth as f32 / max_depth);
    width.max(stroke_px * 0.1).max(0.1)
}

fn stroke_alpha(stroke_opacity: f32, opacity_fade: f32, depth: u16, max_depth: u32) -> u8 {
    let max_depth = (max_depth as f32).max(1.0);
    let alpha = (255.0 * stroke_opacity * (1.0 - opacity_fade * depth as f32 / max_depth))
        .round()
        .max(0.0);
    alpha.min(255.0) as u8
}

fn stroke_tile(
    pixmap: &mut Pixmap,
    segments: &[PxSeg],
    origin_x: f32,
    origin_y: f32,
    stroke_px: f32,
    stroke_opacity: f32,
    opacity_fade: f32,
    stroke_taper: f32,
    max_depth: u32,
) {
    if segments.is_empty() {
        return;
    }
    let mut by_depth: Vec<Vec<PxSeg>> = (0..13).map(|_| Vec::new()).collect();
    for segment in segments {
        let slot = (segment.depth as usize).min(12);
        by_depth[slot].push(PxSeg {
            x0: segment.x0 - origin_x,
            y0: segment.y0 - origin_y,
            x1: segment.x1 - origin_x,
            y1: segment.y1 - origin_y,
            depth: segment.depth,
            color: segment.color,
        });
    }

    let mut stroke = Stroke::default();
    stroke.width = stroke_px;
    stroke.line_cap = LineCap::Round;
    stroke.line_join = LineJoin::Round;

    for group in by_depth {
        if let Some(segment) = group.first() {
            stroke.width = tapered_stroke_width(stroke_px, stroke_taper, segment.depth, max_depth);
        }
        let mut start = 0;
        while start < group.len() {
            let color = group[start].color;
            let mut end = start + 1;
            while end < group.len() && group[end].color == color {
                end += 1;
            }
            let [red, green, blue] = color;
            let alpha = stroke_alpha(stroke_opacity, opacity_fade, group[start].depth, max_depth);
            let mut paint = Paint::default();
            paint.set_color_rgba8(red, green, blue, alpha);
            paint.anti_alias = true;
            for chunk in group[start..end].chunks(PATH_CHUNK) {
                let mut builder = PathBuilder::new();
                for segment in chunk {
                    builder.move_to(segment.x0, segment.y0);
                    builder.line_to(segment.x1, segment.y1);
                }
                let Some(path) = builder.finish() else {
                    continue;
                };
                pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
            }
            start = end;
        }
    }
}

/// Rasterized motifs, keyed by id and revision. One geometry pass fills every tier.
#[derive(Default)]
pub struct StampCache {
    entries: HashMap<(u64, u64), Vec<(u32, Pixmap)>>,
    pub renders: u32,
}

impl StampCache {
    /// Render any motif in `motifs` that is missing a stamp at least `top` pixels wide.
    pub fn ensure_motifs(&mut self, motifs: &[Motif], top: u32) {
        let top = stamp_tier(top as f32, top);
        self.entries.retain(|(id, revision), _| {
            motifs
                .iter()
                .any(|motif| motif.id == *id && motif.revision == *revision)
        });
        let mut seen = HashSet::new();
        for motif in motifs {
            if !seen.insert((motif.id, motif.revision)) {
                continue;
            }
            self.ensure_one(motif, top);
        }
    }

    fn ensure_one(&mut self, motif: &Motif, top: u32) {
        if let Some(tiers) = self.entries.get(&(motif.id, motif.revision)) {
            if tiers.first().is_some_and(|(size, _)| *size >= top) {
                return;
            }
        }
        let Some(stamp) = render_motif_stamp(&motif.params, top) else {
            return;
        };
        self.renders += 1;
        let mut tiers = Vec::new();
        let mut current = stamp;
        let mut size = top;
        loop {
            let next = size / 2;
            if next < 32 {
                tiers.push((size, current));
                break;
            }
            let Some(smaller) = downsample(&current, next) else {
                tiers.push((size, current));
                break;
            };
            tiers.push((size, current));
            current = smaller;
            size = next;
        }
        self.entries.insert((motif.id, motif.revision), tiers);
    }
}

/// Draw each placement from a cached stamp. Identical motifs are rasterized once.
pub fn composite_arrangement(
    cache: &mut StampCache,
    motifs: &[Motif],
    placed: &ArrangementOutput,
    background: [u8; 3],
    bg_mode: BgMode,
    bg_end: [u8; 3],
    bg_angle: f32,
    width: u32,
    height: u32,
    max_tier: u32,
    glow: f32,
    glow_radius: f32,
) -> Option<Pixmap> {
    if width == 0 || height == 0 {
        return None;
    }
    let mut full = Pixmap::new(width, height)?;
    fill_background(&mut full, Some(background), bg_mode, bg_end, bg_angle);
    if placed.placements.is_empty() {
        apply_glow(&mut full, glow, glow_radius);
        return Some(full);
    }

    let bounds = placement_bounds(&placed.placements);
    let view = fit_view(width, height, bounds, 0.0, MARGIN);
    let max_dest = placed
        .placements
        .iter()
        .map(|placement| placement.scale * view.scale)
        .fold(1.0_f32, f32::max);
    let top = stamp_tier(max_dest, max_tier);
    let used_ids: HashSet<u64> = placed.placements.iter().map(|placement| placement.motif_id).collect();
    let used: Vec<Motif> = motifs
        .iter()
        .filter(|motif| used_ids.contains(&motif.id))
        .cloned()
        .collect();
    cache.ensure_motifs(&used, top);

    let mut order: Vec<usize> = (0..placed.placements.len()).collect();
    order.sort_by_key(|&index| placed.placements[index].depth);
    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bilinear;
    paint.blend_mode = BlendMode::SourceOver;
    paint.opacity = 1.0;

    for index in order {
        let placement = placed.placements[index];
        let dest = placement.scale * view.scale;
        if dest < 1.5 {
            continue;
        }
        let Some(motif) = motifs.iter().find(|motif| motif.id == placement.motif_id) else {
            continue;
        };
        let Some(tiers) = cache.entries.get(&(motif.id, motif.revision)) else {
            continue;
        };
        let chosen = pick_tier(tiers, dest);
        let (tier_size, stamp) = &tiers[chosen];
        let (cx, cy) = world_to_px(&view, placement.center.x, placement.center.y);
        let stamp_px = *tier_size as f32;
        let mirror = if placement.reflected { -1.0 } else { 1.0 };
        let mut transform = Transform::identity();
        transform = transform.post_translate(-stamp_px * 0.5, -stamp_px * 0.5);
        transform = transform.post_scale(mirror * dest / stamp_px, dest / stamp_px);
        transform = transform.post_rotate(-placement.angle.to_degrees());
        transform = transform.post_translate(cx, cy);
        full.draw_pixmap(0, 0, stamp.as_ref(), &paint, transform, None);
    }
    apply_glow(&mut full, glow, glow_radius);
    Some(full)
}

fn render_motif_stamp(params: &FractalParams, size: u32) -> Option<Pixmap> {
    let output = crate::geom::generate(params, size, size);
    // Glow is a full-image pass. Stamps stay unbloomed so arrangement glow runs once on the composite.
    render_pixmap_inner(
        &output,
        size,
        size,
        None,
        BgMode::Solid,
        [0, 0, 0],
        0.0,
        0.0,
        params.stroke_opacity,
        params.opacity_fade,
        params.stroke_taper,
        params.depth,
        0.0,
        0.0,
        params,
    )
}

fn downsample(src: &Pixmap, size: u32) -> Option<Pixmap> {
    let mut dst = Pixmap::new(size, size)?;
    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bilinear;
    paint.blend_mode = BlendMode::SourceOver;
    let scale = size as f32 / src.width().max(1) as f32;
    dst.draw_pixmap(
        0,
        0,
        src.as_ref(),
        &paint,
        Transform::from_scale(scale, scale),
        None,
    );
    Some(dst)
}

fn stamp_tier(max_dest_px: f32, max_tier: u32) -> u32 {
    let cap = max_tier.clamp(32, 4096);
    let needed = (max_dest_px.ceil().max(1.0) as u32).clamp(1, cap);
    needed.next_power_of_two().clamp(32, cap.next_power_of_two().min(4096))
}

fn pick_tier(tiers: &[(u32, Pixmap)], dest: f32) -> usize {
    let mut chosen = 0;
    for (index, (size, _)) in tiers.iter().enumerate() {
        if (*size as f32) >= dest {
            chosen = index;
        } else {
            break;
        }
    }
    chosen
}

fn placement_bounds(placements: &[StampPlacement]) -> Bounds {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for placement in placements {
        let half = placement.scale * 0.5;
        let expand = half * (placement.angle.cos().abs() + placement.angle.sin().abs());
        min_x = min_x.min(placement.center.x - expand);
        min_y = min_y.min(placement.center.y - expand);
        max_x = max_x.max(placement.center.x + expand);
        max_y = max_y.max(placement.center.y + expand);
    }
    if !min_x.is_finite() {
        return Bounds {
            min_x: -0.5,
            min_y: -0.5,
            max_x: 0.5,
            max_y: 0.5,
        };
    }
    Bounds {
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

/// Additive bloom. `glow <= 0` returns without touching pixels.
fn apply_glow(pixmap: &mut Pixmap, glow: f32, glow_radius: f32) {
    if !(glow.is_finite() && glow > 0.0) {
        return;
    }
    let radius = if glow_radius.is_finite() {
        glow_radius.clamp(0.0, MAX_GLOW_RADIUS)
    } else {
        0.0
    };
    let width = pixmap.width() as usize;
    let height = pixmap.height() as usize;
    if width == 0 || height == 0 {
        return;
    }

    let original = pixmap.data().to_vec();
    let mut bright = Vec::with_capacity(width * height * 3);
    for px in original.chunks_exact(4) {
        let red = px[0] as f32;
        let green = px[1] as f32;
        let blue = px[2] as f32;
        let luma = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0;
        bright.push(red * luma);
        bright.push(green * luma);
        bright.push(blue * luma);
    }

    let kernel = gaussian_kernel(radius);
    let mut blurred = vec![0.0_f32; bright.len()];
    blur_horizontal(&bright, &mut blurred, width, height, &kernel);
    blur_vertical(&blurred, &mut bright, width, height, &kernel);

    let data = pixmap.data_mut();
    for (index, px) in data.chunks_exact_mut(4).enumerate() {
        let src = index * 4;
        let base = index * 3;
        for channel in 0..3 {
            let summed = original[src + channel] as f32 + bright[base + channel] * glow;
            px[channel] = summed.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn gaussian_kernel(radius: f32) -> Vec<f32> {
    let half = radius.ceil() as i32;
    if half <= 0 {
        return vec![1.0];
    }
    let half = half.min(MAX_GLOW_RADIUS as i32);
    let sigma = (radius / 2.0).max(0.5);
    let mut weights = Vec::with_capacity((half * 2 + 1) as usize);
    let mut sum = 0.0_f32;
    for offset in -half..=half {
        let x = offset as f32;
        let weight = (-0.5 * x * x / (sigma * sigma)).exp();
        weights.push(weight);
        sum += weight;
    }
    let inv = 1.0 / sum;
    for weight in &mut weights {
        *weight *= inv;
    }
    weights
}

fn blur_horizontal(src: &[f32], dst: &mut [f32], width: usize, height: usize, kernel: &[f32]) {
    let half = (kernel.len() / 2) as isize;
    let max_x = width as isize - 1;
    dst.par_chunks_mut(width * 3)
        .enumerate()
        .take(height)
        .for_each(|(y, out_row)| {
            let in_row = &src[y * width * 3..(y + 1) * width * 3];
            for x in 0..width {
                let mut acc = [0.0_f32; 3];
                for (k, &weight) in kernel.iter().enumerate() {
                    let sample_x = (x as isize + k as isize - half).clamp(0, max_x) as usize;
                    let sample = sample_x * 3;
                    acc[0] += in_row[sample] * weight;
                    acc[1] += in_row[sample + 1] * weight;
                    acc[2] += in_row[sample + 2] * weight;
                }
                let dest = x * 3;
                out_row[dest] = acc[0];
                out_row[dest + 1] = acc[1];
                out_row[dest + 2] = acc[2];
            }
        });
}

fn blur_vertical(src: &[f32], dst: &mut [f32], width: usize, height: usize, kernel: &[f32]) {
    let half = (kernel.len() / 2) as isize;
    let max_y = height as isize - 1;
    dst.par_chunks_mut(width * 3)
        .enumerate()
        .take(height)
        .for_each(|(y, out_row)| {
            for x in 0..width {
                let mut acc = [0.0_f32; 3];
                for (k, &weight) in kernel.iter().enumerate() {
                    let sample_y = (y as isize + k as isize - half).clamp(0, max_y) as usize;
                    let sample = (sample_y * width + x) * 3;
                    acc[0] += src[sample] * weight;
                    acc[1] += src[sample + 1] * weight;
                    acc[2] += src[sample + 2] * weight;
                }
                let dest = x * 3;
                out_row[dest] = acc[0];
                out_row[dest + 1] = acc[1];
                out_row[dest + 2] = acc[2];
            }
        });
}

fn copy_tile_from(dst: &mut Pixmap, src: &Pixmap, origin_x: u32, origin_y: u32) {
    let src_w = src.width();
    let dst_w = dst.width();
    let dst_h = dst.height();
    let src_data = src.data();
    let dst_data = dst.data_mut();
    for row in 0..dst_h {
        let src_off = (((origin_y + row) * src_w + origin_x) * 4) as usize;
        let dst_off = (row * dst_w * 4) as usize;
        let bytes = (dst_w * 4) as usize;
        dst_data[dst_off..dst_off + bytes].copy_from_slice(&src_data[src_off..src_off + bytes]);
    }
}

fn blit(dst: &mut Pixmap, origin_x: u32, origin_y: u32, src: &Pixmap) {
    let dst_w = dst.width();
    let src_w = src.width();
    let src_h = src.height();
    let dst_data = dst.data_mut();
    let src_data = src.data();
    for row in 0..src_h {
        let dst_off = (((origin_y + row) * dst_w + origin_x) * 4) as usize;
        let src_off = (row * src_w * 4) as usize;
        let bytes = (src_w * 4) as usize;
        dst_data[dst_off..dst_off + bytes].copy_from_slice(&src_data[src_off..src_off + bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{generate, ColourMode, FillMode, FractalParams};
    use std::env;

    #[test]
    fn small_render_has_background_and_strokes() {
        let params = FractalParams::default();
        let output = generate(&params, 64, 64);
        let pixmap = render_pixmap(&output, &params, 64, 64).expect("pixmap");
        assert_eq!(pixmap.width(), 64);
        assert_eq!(pixmap.height(), 64);
        let pixels = pixmap.data();
        let bg = params.background;
        assert_eq!(pixels[0], bg[0]);
        assert_eq!(pixels[1], bg[1]);
        assert_eq!(pixels[2], bg[2]);
        assert_eq!(pixels[3], 255);
        let painted = pixels.chunks_exact(4).any(|px| {
            px[0] != bg[0] || px[1] != bg[1] || px[2] != bg[2]
        });
        assert!(painted, "expected stroke pixels");
    }

    #[test]
    fn png_roundtrip_keeps_dimensions() {
        let params = FractalParams::default();
        let output = generate(&params, 32, 48);
        let pixmap = render_pixmap(&output, &params, 32, 48).expect("pixmap");
        let path = env::temp_dir().join("polygon-fractal-test.png");
        write_png(&path, &pixmap).expect("write png");
        let file = File::open(&path).expect("reopen");
        let decoder = png::Decoder::new(file);
        let reader = decoder.read_info().expect("decode");
        assert_eq!(reader.info().width, 32);
        assert_eq!(reader.info().height, 48);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn svg_export_writes_strokes_for_default_params() {
        let params = FractalParams::default();
        let output = generate(&params, 64, 64);
        assert!(!output.segments.is_empty());
        let path = env::temp_dir().join("polygon-fractal-test.svg");
        write_svg(&path, &output, &params, 64, 64).expect("write svg");
        let text = std::fs::read_to_string(&path).expect("read svg");
        let len = std::fs::metadata(&path).expect("stat svg").len();
        let _ = std::fs::remove_file(&path);
        assert!(len > 0, "svg file was empty");
        assert!(!text.is_empty());
        assert!(text.contains("<svg"), "missing svg root");
        assert!(
            text.contains("<path") || text.contains("<line"),
            "missing stroke element"
        );
    }

    #[test]
    fn stamp_keeps_transparent_background_and_ink() {
        let params = FractalParams::default();
        let output = generate(&params, 48, 48);
        let pixmap = render_pixmap_inner(
            &output,
            48,
            48,
            None,
            BgMode::Solid,
            [0, 0, 0],
            0.0,
            0.0,
            params.stroke_opacity,
            params.opacity_fade,
            params.stroke_taper,
            params.depth,
            0.0,
            0.0,
            &params,
        )
        .expect("stamp");
        let pixels = pixmap.data();
        assert!(pixels.chunks_exact(4).any(|px| px[3] == 0));
        assert!(pixels.chunks_exact(4).any(|px| px[3] > 0));
    }

    #[test]
    fn shared_motif_is_rasterized_once_and_a_clone_is_not() {
        let params = FractalParams::default();
        let motif = Motif {
            id: 1,
            name: "A".to_string(),
            revision: 1,
            params: params.clone(),
        };
        let clone = Motif {
            id: 2,
            name: "B".to_string(),
            revision: 2,
            params,
        };
        let mut cache = StampCache::default();
        cache.ensure_motifs(&[motif.clone(), motif.clone()], 64);
        assert_eq!(cache.renders, 1);

        let mut arrangement = crate::geom::ArrangementParams::default();
        arrangement.depth = 1;
        arrangement.motif_by_depth = vec![1, 1];
        let placed = crate::geom::arrange(&arrangement, 128, 128);
        assert!(placed.placements.len() >= 2);
        let pixmap = composite_arrangement(
            &mut cache,
            &[motif.clone()],
            &placed,
            [0, 0, 0],
            BgMode::Solid,
            [0, 0, 0],
            0.0,
            128,
            128,
            64,
            0.0,
            0.0,
        )
        .expect("composite");
        assert_eq!(pixmap.width(), 128);
        assert_eq!(cache.renders, 1);
        let pixels = pixmap.data();
        assert!(pixels.chunks_exact(4).any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0));

        arrangement.motif_by_depth = vec![1, 2];
        let placed = crate::geom::arrange(&arrangement, 128, 128);
        let _ = composite_arrangement(
            &mut cache,
            &[motif, clone],
            &placed,
            [0, 0, 0],
            BgMode::Solid,
            [0, 0, 0],
            0.0,
            128,
            128,
            64,
            0.0,
            0.0,
        );
        assert_eq!(cache.renders, 2);
    }

    #[test]
    fn default_stroke_alpha_stays_opaque_and_opacity_changes_pixels() {
        let params = FractalParams::default();
        for depth in 0..=params.depth {
            assert_eq!(
                stroke_alpha(params.stroke_opacity, params.opacity_fade, depth as u16, params.depth),
                255,
                "depth {depth}"
            );
        }
        assert_eq!(stroke_alpha(1.0, 1.0, 0, params.depth), 255);
        assert_eq!(stroke_alpha(1.0, 1.0, params.depth as u16, params.depth), 0);
        assert_eq!(stroke_alpha(0.5, 0.0, 3, params.depth), (255.0_f32 * 0.5).round() as u8);

        let output = generate(&params, 64, 64);
        let opaque = render_pixmap(&output, &params, 64, 64).expect("opaque");
        let mut faded = params.clone();
        faded.stroke_opacity = 0.25;
        faded.opacity_fade = 0.0;
        let dim = render_pixmap(&output, &faded, 64, 64).expect("dim");
        assert_ne!(opaque.data(), dim.data());
        assert_eq!(stroke_alpha(faded.stroke_opacity, faded.opacity_fade, 0, faded.depth), 64);
    }

    #[test]
    fn zero_taper_keeps_uniform_width_and_taper_thins_deeper_strokes() {
        let stroke_px = 4.0;
        let max_depth = 6;
        for depth in 0..=max_depth {
            assert_eq!(
                tapered_stroke_width(stroke_px, 0.0, depth as u16, max_depth),
                stroke_px,
                "depth {depth}"
            );
        }
        assert_eq!(tapered_stroke_width(stroke_px, 1.0, 0, max_depth), stroke_px);
        assert_eq!(tapered_stroke_width(stroke_px, 1.0, max_depth as u16, max_depth), stroke_px * 0.1);
        assert_eq!(tapered_stroke_width(0.5, 1.0, max_depth as u16, max_depth), 0.1);

        let params = FractalParams::default();
        let output = generate(&params, 64, 64);
        let uniform = render_pixmap(&output, &params, 64, 64).expect("uniform");
        let mut explicit = params.clone();
        explicit.stroke_taper = 0.0;
        let explicit_uniform = render_pixmap(&output, &explicit, 64, 64).expect("explicit");
        assert_eq!(uniform.data(), explicit_uniform.data());

        let mut tapered = params;
        tapered.stroke_taper = 1.0;
        let thinned = render_pixmap(&output, &tapered, 64, 64).expect("tapered");
        assert_ne!(uniform.data(), thinned.data());
    }

    #[test]
    fn solid_background_fills_corner_and_gradients_vary() {
        let mut params = FractalParams::default();
        params.background = [12, 34, 56];
        params.bg_mode = BgMode::Solid;
        params.bg_end = [200, 10, 10];
        params.bg_angle = 35.0;
        let output = generate(&params, 48, 48);
        let solid = render_pixmap(&output, &params, 48, 48).expect("solid");
        assert_eq!(&solid.data()[0..4], &[12, 34, 56, 255]);

        params.stroke_opacity = 0.0;
        params.background = [255, 0, 0];
        params.bg_end = [0, 0, 255];
        params.bg_angle = 0.0;
        params.bg_mode = BgMode::Linear;
        let linear = render_pixmap(&output, &params, 48, 48).expect("linear");
        assert!(
            row_colour_count(&linear, 0) >= 2,
            "linear background should use at least two colours"
        );

        params.bg_mode = BgMode::Radial;
        let radial = render_pixmap(&output, &params, 48, 48).expect("radial");
        assert!(
            row_colour_count(&radial, 0) >= 2,
            "radial background should use at least two colours"
        );

        let mut tile = Pixmap::new(8, 8).expect("tile");
        paint_background_gradient(
            &mut tile,
            64.0,
            8.0,
            [255, 0, 0],
            BgMode::Linear,
            [0, 0, 255],
            0.0,
            56.0,
            0.0,
        );
        let px = tile.data();
        assert!(
            px[2] > px[0],
            "a tile offset along the gradient should sample the end colour"
        );

        let placed = crate::geom::ArrangementOutput {
            placements: Vec::new(),
            truncated: false,
        };
        let mut cache = StampCache::default();
        let arranged = composite_arrangement(
            &mut cache,
            &[],
            &placed,
            [12, 34, 56],
            BgMode::Solid,
            [0, 0, 0],
            0.0,
            16,
            16,
            64,
            0.0,
            0.0,
        )
        .expect("solid arrangement");
        assert_eq!(&arranged.data()[0..4], &[12, 34, 56, 255]);

        let arranged = composite_arrangement(
            &mut cache,
            &[],
            &placed,
            [255, 0, 0],
            BgMode::Linear,
            [0, 0, 255],
            0.0,
            32,
            16,
            64,
            0.0,
            0.0,
        )
        .expect("linear arrangement");
        assert!(row_colour_count(&arranged, 0) >= 2);

        let mut svg_params = params.clone();
        svg_params.bg_mode = BgMode::Linear;
        let path = env::temp_dir().join("polygon-fractal-gradient.svg");
        write_svg(&path, &output, &svg_params, 48, 48).expect("svg");
        let text = std::fs::read_to_string(&path).expect("read svg");
        let _ = std::fs::remove_file(&path);
        assert!(text.contains("<linearGradient"), "svg should paint the linear background");
    }

    #[test]
    fn zero_glow_leaves_pixels_and_positive_glow_bleeds() {
        let mut params = FractalParams::default();
        assert_eq!(params.glow, 0.0);
        let output = generate(&params, 64, 64);
        let plain = render_pixmap(&output, &params, 64, 64).expect("plain");
        let before = plain.data().to_vec();

        let mut held = plain;
        apply_glow(&mut held, 0.0, 64.0);
        apply_glow(&mut held, -1.0, 32.0);
        apply_glow(&mut held, f32::NAN, 8.0);
        assert_eq!(held.data(), before.as_slice(), "glow <= 0 must not change pixels");

        params.glow_radius = 64.0;
        let wide_off = render_pixmap(&output, &params, 64, 64).expect("wide off");
        assert_eq!(wide_off.data(), before.as_slice());

        params.glow = 1.0;
        params.glow_radius = 16.0;
        let glowed = render_pixmap(&output, &params, 64, 64).expect("glowed");
        assert!(
            background_adjacent_changed(&before, glowed.data(), 64, 64, params.background),
            "glow should change a background pixel next to a stroke"
        );
    }

    fn background_adjacent_changed(
        before: &[u8],
        after: &[u8],
        width: u32,
        height: u32,
        background: [u8; 3],
    ) -> bool {
        let width = width as i32;
        let height = height as i32;
        let pixel = |buf: &[u8], x: i32, y: i32| {
            let index = ((y * width + x) * 4) as usize;
            [buf[index], buf[index + 1], buf[index + 2]]
        };
        let is_bg = |px: [u8; 3]| px == background;
        for y in 0..height {
            for x in 0..width {
                if !is_bg(pixel(before, x, y)) {
                    continue;
                }
                let mut near_ink = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || ny < 0 || nx >= width || ny >= height {
                            continue;
                        }
                        if !is_bg(pixel(before, nx, ny)) {
                            near_ink = true;
                        }
                    }
                }
                if near_ink && pixel(before, x, y) != pixel(after, x, y) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn fill_none_matches_the_stroke_only_render_and_solid_paints_interior() {
        let params = FractalParams::default();
        assert_eq!(params.fill_mode, FillMode::None);
        let output = generate(&params, 64, 64);
        assert!(output.outlines.is_empty(), "fill none must not emit outlines");
        assert_eq!(output.segments.len(), 282);
        let pixmap = render_pixmap(&output, &params, 64, 64).expect("pixmap");
        // Captured from the stroke-only renderer before fills existed.
        assert_eq!(fnv_pixmap(&pixmap), 15_736_216_473_585_605_190);

        let mut faded = params.clone();
        faded.fill_opacity = 0.0;
        let faded_output = generate(&faded, 64, 64);
        assert!(faded_output.outlines.is_empty());
        let faded_px = render_pixmap(&faded_output, &faded, 64, 64).expect("faded");
        assert_eq!(pixmap.data(), faded_px.data());

        let mut solid = params.clone();
        solid.depth = 2;
        solid.background = [0, 0, 0];
        solid.color_start = [255, 0, 0];
        solid.colour_mode = ColourMode::Single;
        solid.fill_mode = FillMode::None;
        let open = generate(&solid, 128, 128);
        assert!(open.outlines.is_empty());
        let open_px = render_pixmap(&open, &solid, 128, 128).expect("open");

        solid.fill_mode = FillMode::Solid;
        solid.fill_opacity = 1.0;
        let filled = generate(&solid, 128, 128);
        assert!(!filled.outlines.is_empty());
        assert_eq!(open.segments.len(), filled.segments.len());
        for (before, after) in open.segments.iter().zip(&filled.segments) {
            assert_eq!(before.x0, after.x0);
            assert_eq!(before.y0, after.y0);
            assert_eq!(before.x1, after.x1);
            assert_eq!(before.y1, after.y1);
            assert_eq!(before.depth, after.depth);
            assert_eq!(before.color, after.color);
        }
        let filled_px = render_pixmap(&filled, &solid, 128, 128).expect("filled");
        let painted = open_px
            .data()
            .chunks_exact(4)
            .zip(filled_px.data().chunks_exact(4))
            .filter(|(before, after)| before[..3] == [0, 0, 0] && after[..3] == [255, 0, 0])
            .count();
        assert!(
            painted > 30,
            "solid fill should cover interior pixels that were background, got {painted}"
        );

        solid.fill_mode = FillMode::Palette;
        solid.palette = vec![[0, 255, 0], [0, 0, 255]];
        solid.gradient_hsv = false;
        let palette_px = render_pixmap(
            &generate(&solid, 128, 128),
            &solid,
            128,
            128,
        )
        .expect("palette");
        let green = open_px
            .data()
            .chunks_exact(4)
            .zip(palette_px.data().chunks_exact(4))
            .filter(|(before, after)| before[..3] == [0, 0, 0] && after[..3] == [0, 255, 0])
            .count();
        assert!(green > 30, "palette fill should sample the depth-0 stop, got {green}");
    }

    fn fnv_pixmap(pixmap: &Pixmap) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in pixmap.data() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    fn row_colour_count(pixmap: &Pixmap, y: u32) -> usize {
        let mut colours = std::collections::HashSet::new();
        let width = pixmap.width();
        let data = pixmap.data();
        for x in 0..width {
            let index = ((y * width + x) * 4) as usize;
            colours.insert([data[index], data[index + 1], data[index + 2]]);
        }
        colours.len()
    }
}

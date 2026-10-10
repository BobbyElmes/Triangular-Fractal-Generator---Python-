//! Anti-aliased outline rendering and PNG export.
//!
//! Preview and export share one path: segments are binned into tiles and
//! stroked in parallel, then copied into a single pixmap.

use crate::geom::{Bounds, FractalParams, Output, Segment};
use png::{BitDepth, ColorType, Encoder};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tiny_skia::{Color, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform};

const TILE: u32 = 512;
const MARGIN: f32 = 0.04;
const PATH_CHUNK: usize = 8_000;

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
    if width == 0 || height == 0 {
        return None;
    }
    let mut full = Pixmap::new(width, height)?;
    let bg = params.background;
    full.fill(Color::from_rgba8(bg[0], bg[1], bg[2], 255));

    let view = fit_view(width, height, output.bounds, output.stroke_world);
    let stroke_px = (output.stroke_world * view.scale).max(0.25);
    let px_segs = pixel_segments(&output.segments, &view);

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
                tile.fill(Color::from_rgba8(bg[0], bg[1], bg[2], 255));
                stroke_tile(
                    &mut tile,
                    &bins[index],
                    origin_x as f32,
                    origin_y as f32,
                    stroke_px,
                );
                Some((origin_x, origin_y, tile))
            })
            .collect();
        for (origin_x, origin_y, tile) in rendered {
            blit(&mut full, origin_x, origin_y, &tile);
        }
    }

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

fn fit_view(width: u32, height: u32, bounds: Bounds, stroke_world: f32) -> View {
    let pad = stroke_world * 0.5;
    let min_x = bounds.min_x - pad;
    let min_y = bounds.min_y - pad;
    let box_w = (bounds.max_x - bounds.min_x + stroke_world).max(1e-4);
    let box_h = (bounds.max_y - bounds.min_y + stroke_world).max(1e-4);
    let avail_w = width as f32 * (1.0 - 2.0 * MARGIN);
    let avail_h = height as f32 * (1.0 - 2.0 * MARGIN);
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

fn world_to_px(view: &View, x: f32, y: f32) -> (f32, f32) {
    let px = x * view.scale + view.tx;
    let py = view.height - (y * view.scale + view.ty);
    (px, py)
}

fn stroke_tile(
    pixmap: &mut Pixmap,
    segments: &[PxSeg],
    origin_x: f32,
    origin_y: f32,
    stroke_px: f32,
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
        let mut start = 0;
        while start < group.len() {
            let color = group[start].color;
            let mut end = start + 1;
            while end < group.len() && group[end].color == color {
                end += 1;
            }
            let [red, green, blue] = color;
            let mut paint = Paint::default();
            paint.set_color_rgba8(red, green, blue, 255);
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
    use crate::geom::{generate, FractalParams};
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
}

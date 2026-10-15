//! Stdio MCP server for the polygon fractal engine.
//!
//! Generation and rasterization stay in the `polygon_fractal` library. This
//! binary only translates tool arguments, calls those functions, and writes
//! the files.

mod dto;

use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine;
use polygon_fractal::geom::{arrange, generate, FractalParams, Motif};
use polygon_fractal::render::{
    composite_arrangement, render_pixmap, write_png, write_svg, StampCache, MAX_STAMP_TIER,
};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Serialize;

use crate::dto::{
    motif_ids, ImageFormat, RandomizeShapeRequest, RenderArrangementRequest, RenderShapeRequest,
    ResolvedArrangementParams, ResolvedShapeParams,
};

const MIN_PX: u32 = 16;
const MAX_PX: u32 = 16_384;

#[derive(Debug, Clone)]
struct FractalServer;

#[tool_router]
impl FractalServer {
    /// Render a polygon edge-fractal (shape mode) to PNG or SVG.
    ///
    /// Every shape field is optional. Omitted fields use the engine defaults, then
    /// `FractalParams::sanitized` clamps them to the same ranges as the GUI.
    /// `width` and `height` are pixels clamped to 16..=16384. `format` is `png` or `svg`.
    /// Set `return_image` to also receive the file inline (PNG as an image, SVG as text).
    /// The result echoes the resolved params, the segment count, and whether generation
    /// hit the segment budget (`truncated`).
    #[tool(name = "render_shape_fractal")]
    async fn render_shape_fractal(
        &self,
        Parameters(request): Parameters<RenderShapeRequest>,
    ) -> Result<CallToolResult, String> {
        tokio::task::spawn_blocking(move || render_shape_request(request))
            .await
            .map_err(|err| format!("render task failed: {err}"))?
    }

    /// Stamp shape motifs with the arrangement growth rule and write a PNG.
    ///
    /// SVG is rejected; the engine only writes SVG for shape fractals. `motifs` is a
    /// list of `{ name?, params }`. `motif_by_depth` lists indices into `motifs`, one
    /// per depth from 0 through `depth`. Motif ids are assigned as index + 1. A short
    /// index list repeats its last entry. Out-of-range indices are an error. `width`
    /// and `height` are clamped to 16..=16384. The result reports placement count as
    /// `placements`, plus the resolved arrangement and motif params.
    #[tool(name = "render_arrangement_fractal")]
    async fn render_arrangement_fractal(
        &self,
        Parameters(request): Parameters<RenderArrangementRequest>,
    ) -> Result<CallToolResult, String> {
        tokio::task::spawn_blocking(move || render_arrangement_request(request))
            .await
            .map_err(|err| format!("render task failed: {err}"))?
    }

    /// Build a random valid shape fractal from `seed`, render it, and return the resolved params.
    ///
    /// The same seed repeats the same parameters. Use those params with
    /// `render_shape_fractal` to iterate. `format` is `png` or `svg`. `width` and
    /// `height` are clamped to 16..=16384.
    #[tool(name = "randomize_shape_fractal")]
    async fn randomize_shape_fractal(
        &self,
        Parameters(request): Parameters<RandomizeShapeRequest>,
    ) -> Result<CallToolResult, String> {
        tokio::task::spawn_blocking(move || render_random_shape(request))
            .await
            .map_err(|err| format!("render task failed: {err}"))?
    }
}

#[tool_handler(
    name = "polygon-fractal",
    instructions = "Render polygon fractals to PNG or SVG. Shape tools take the full option set. Arrangement tools stamp motifs and write PNG only. Omitted fields use engine defaults, then values are clamped the same way as the GUI."
)]
impl ServerHandler for FractalServer {}

#[derive(Serialize)]
struct ShapeRenderSummary {
    path: String,
    format: &'static str,
    width: u32,
    height: u32,
    segments: usize,
    truncated: bool,
    elapsed_ms: f64,
    params: ResolvedShapeParams,
}

#[derive(Serialize)]
struct RandomizeSummary {
    seed: u64,
    path: String,
    format: &'static str,
    width: u32,
    height: u32,
    segments: usize,
    truncated: bool,
    elapsed_ms: f64,
    params: ResolvedShapeParams,
}

#[derive(Serialize)]
struct ResolvedMotif {
    index: usize,
    name: String,
    params: ResolvedShapeParams,
}

#[derive(Serialize)]
struct ArrangementRenderSummary {
    path: String,
    format: &'static str,
    width: u32,
    height: u32,
    placements: usize,
    truncated: bool,
    elapsed_ms: f64,
    params: ResolvedArrangementParams,
    motifs: Vec<ResolvedMotif>,
}

fn render_shape_request(request: RenderShapeRequest) -> Result<CallToolResult, String> {
    let params = request.shape.into_params();
    let (width, height) = clamp_size(request.width, request.height);
    let started = Instant::now();
    let rendered = render_shape(
        &params,
        width,
        height,
        request.format,
        &request.output_path,
        request.return_image,
    )?;
    let summary = ShapeRenderSummary {
        path: request.output_path,
        format: request.format.as_str(),
        width,
        height,
        segments: rendered.segments,
        truncated: rendered.truncated,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        params: ResolvedShapeParams::from(&params),
    };
    finish(summary, rendered.inline)
}

fn render_random_shape(request: RandomizeShapeRequest) -> Result<CallToolResult, String> {
    let mut params = FractalParams::default();
    params.randomize(request.seed);
    let (width, height) = clamp_size(request.width, request.height);
    let started = Instant::now();
    let rendered = render_shape(
        &params,
        width,
        height,
        request.format,
        &request.output_path,
        false,
    )?;
    let summary = RandomizeSummary {
        seed: request.seed,
        path: request.output_path,
        format: request.format.as_str(),
        width,
        height,
        segments: rendered.segments,
        truncated: rendered.truncated,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        params: ResolvedShapeParams::from(&params),
    };
    finish(summary, rendered.inline)
}

fn render_arrangement_request(request: RenderArrangementRequest) -> Result<CallToolResult, String> {
    if request.format == Some(ImageFormat::Svg) {
        return Err(
            "SVG export is only available in Shape mode. render_arrangement_fractal writes PNG only."
                .to_string(),
        );
    }
    if request.motifs.is_empty() {
        return Err("at least one motif is required".to_string());
    }
    let ids = motif_ids(&request.motif_by_depth, request.motifs.len())?;
    let motifs: Vec<Motif> = request
        .motifs
        .iter()
        .enumerate()
        .map(|(index, motif)| {
            let id = index as u64 + 1;
            let name = motif
                .name
                .as_ref()
                .map(|name| name.trim())
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("Motif {id}"));
            Motif {
                id,
                name,
                revision: 1,
                params: motif.params.clone().into_params(),
            }
        })
        .collect();
    let arrangement = request.arrangement.into_params(ids);
    let (width, height) = clamp_size(request.width, request.height);
    let path = prepare_output(&request.output_path)?;
    let started = Instant::now();
    let placed = arrange(&arrangement, width, height);
    let mut cache = StampCache::default();
    let pixmap = composite_arrangement(
        &mut cache,
        &motifs,
        &placed,
        arrangement.background,
        arrangement.bg_mode,
        arrangement.bg_end,
        arrangement.bg_angle,
        width,
        height,
        MAX_STAMP_TIER,
        arrangement.glow,
        arrangement.glow_radius,
    )
    .ok_or_else(|| "could not allocate the image".to_string())?;
    write_png(&path, &pixmap)?;
    let inline = if request.return_image {
        Some(inline_png(&path)?)
    } else {
        None
    };
    let summary = ArrangementRenderSummary {
        path: request.output_path,
        format: "png",
        width,
        height,
        placements: placed.placements.len(),
        truncated: placed.truncated,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        params: ResolvedArrangementParams::from(&arrangement),
        motifs: motifs
            .iter()
            .map(|motif| ResolvedMotif {
                index: motif.id.saturating_sub(1) as usize,
                name: motif.name.clone(),
                params: ResolvedShapeParams::from(&motif.params),
            })
            .collect(),
    };
    finish(summary, inline)
}

struct RenderedShape {
    segments: usize,
    truncated: bool,
    inline: Option<ContentBlock>,
}

fn render_shape(
    params: &FractalParams,
    width: u32,
    height: u32,
    format: ImageFormat,
    output_path: &str,
    return_image: bool,
) -> Result<RenderedShape, String> {
    let path = prepare_output(output_path)?;
    let output = generate(params, width, height);
    match format {
        ImageFormat::Png => {
            let pixmap = render_pixmap(&output, params, width, height)
                .ok_or_else(|| "could not allocate the image".to_string())?;
            write_png(&path, &pixmap)?;
        }
        ImageFormat::Svg => {
            write_svg(&path, &output, params, width, height)?;
        }
    }
    let inline = if return_image {
        Some(match format {
            ImageFormat::Png => inline_png(&path)?,
            ImageFormat::Svg => inline_svg(&path)?,
        })
    } else {
        None
    };
    Ok(RenderedShape {
        segments: output.segments.len(),
        truncated: output.truncated,
        inline,
    })
}

fn clamp_size(width: u32, height: u32) -> (u32, u32) {
    (width.clamp(MIN_PX, MAX_PX), height.clamp(MIN_PX, MAX_PX))
}

fn prepare_output(output_path: &str) -> Result<PathBuf, String> {
    if output_path.trim().is_empty() {
        return Err("output_path must not be empty".to_string());
    }
    let path = PathBuf::from(output_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("could not create output directory: {err}"))?;
        }
    }
    Ok(path)
}

fn inline_png(path: &Path) -> Result<ContentBlock, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("could not read {path:?}: {err}"))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(ContentBlock::image(data, "image/png"))
}

fn inline_svg(path: &Path) -> Result<ContentBlock, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("could not read {path:?}: {err}"))?;
    Ok(ContentBlock::text(text))
}

fn finish(summary: impl Serialize, extra: Option<ContentBlock>) -> Result<CallToolResult, String> {
    let value = serde_json::to_value(&summary)
        .map_err(|err| format!("could not encode the result: {err}"))?;
    let pretty = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("could not encode the result: {err}"))?;
    let mut content = vec![ContentBlock::text(pretty)];
    if let Some(block) = extra {
        content.push(block);
    }
    let mut result = CallToolResult::structured(value);
    result.content = content;
    Ok(result)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let running = FractalServer.serve(stdio()).await?;
    running.waiting().await?;
    Ok(())
}

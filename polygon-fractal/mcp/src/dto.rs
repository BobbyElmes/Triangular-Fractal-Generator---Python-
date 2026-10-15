//! Request DTOs for the MCP tools.
//!
//! Every shape and arrangement field is optional. Conversion starts from the
//! engine defaults, applies whatever the caller sent, then uses
//! [`FractalParams::sanitized`] / [`ArrangementParams::sanitized`] so the GUI
//! and this server share one set of bounds.

use polygon_fractal::geom::{ArrangementParams, BgMode, ColourMode, FillMode, FractalParams};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `single`, `gradient`, `random`, or `palette`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ColourModeDto {
    Single,
    Gradient,
    Random,
    Palette,
}

/// `solid`, `linear`, or `radial`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BgModeDto {
    Solid,
    Linear,
    Radial,
}

/// `none`, `solid`, or `palette`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FillModeDto {
    None,
    Solid,
    Palette,
}

/// `png` or `svg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Svg,
}

impl ImageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Svg => "svg",
        }
    }
}

impl From<ColourModeDto> for ColourMode {
    fn from(value: ColourModeDto) -> Self {
        match value {
            ColourModeDto::Single => Self::Single,
            ColourModeDto::Gradient => Self::Gradient,
            ColourModeDto::Random => Self::Random,
            ColourModeDto::Palette => Self::Palette,
        }
    }
}

impl From<ColourMode> for ColourModeDto {
    fn from(value: ColourMode) -> Self {
        match value {
            ColourMode::Single => Self::Single,
            ColourMode::Gradient => Self::Gradient,
            ColourMode::Random => Self::Random,
            ColourMode::Palette => Self::Palette,
        }
    }
}

impl From<BgModeDto> for BgMode {
    fn from(value: BgModeDto) -> Self {
        match value {
            BgModeDto::Solid => Self::Solid,
            BgModeDto::Linear => Self::Linear,
            BgModeDto::Radial => Self::Radial,
        }
    }
}

impl From<BgMode> for BgModeDto {
    fn from(value: BgMode) -> Self {
        match value {
            BgMode::Solid => Self::Solid,
            BgMode::Linear => Self::Linear,
            BgMode::Radial => Self::Radial,
        }
    }
}

impl From<FillModeDto> for FillMode {
    fn from(value: FillModeDto) -> Self {
        match value {
            FillModeDto::None => Self::None,
            FillModeDto::Solid => Self::Solid,
            FillModeDto::Palette => Self::Palette,
        }
    }
}

impl From<FillMode> for FillModeDto {
    fn from(value: FillMode) -> Self {
        match value {
            FillMode::None => Self::None,
            FillMode::Solid => Self::Solid,
            FillMode::Palette => Self::Palette,
        }
    }
}

/// Full shape-fractal option set. Omitted fields keep the engine defaults, then every value is clamped.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ShapeParamsDto {
    /// Seed polygon side lengths. The count is clamped to 3..=16 (shorter lists are padded with the average length, longer lists are truncated). Each length is clamped to 0.05..=8.
    #[serde(default)]
    pub side_lengths: Option<Vec<f32>>,
    /// Child size as a fraction of its parent. Clamped to 0.05..=0.95.
    #[serde(default)]
    pub shrink: Option<f32>,
    /// Per-depth shrink factors, each clamped to 0.05..=0.95. Empty or omitted uses `shrink` at every depth. Entry `d` is the factor for a child created at depth `d`.
    #[serde(default)]
    pub shrink_by_depth: Option<Vec<f32>>,
    /// Recursion depth. Clamped to 0..=16. 0 draws only the seed polygon.
    #[serde(default)]
    pub depth: Option<u32>,
    /// Root rotation in degrees. Non-finite values are reset to 0.
    #[serde(default)]
    pub rotation_deg: Option<f32>,
    /// When true, children grow toward the parent centroid.
    #[serde(default)]
    pub inward: Option<bool>,
    /// When true, skip a child whose body points back toward the origin. The seed polygon is always drawn.
    #[serde(default)]
    pub outward_only: Option<bool>,
    /// Extra rotation applied to each child, in degrees. Non-finite values are reset to 0.
    #[serde(default)]
    pub twist_deg: Option<f32>,
    /// How outlines are coloured: `single`, `gradient`, `random`, or `palette`.
    #[serde(default)]
    pub colour_mode: Option<ColourModeDto>,
    /// RGB start colour, each channel 0..=255. Used by single, gradient, and solid fill.
    #[serde(default)]
    pub color_start: Option<[u8; 3]>,
    /// RGB end colour, each channel 0..=255. Used by gradient mode.
    #[serde(default)]
    pub color_end: Option<[u8; 3]>,
    /// When true, gradient and palette modes blend in HSV (shortest hue). False blends per RGB channel.
    #[serde(default)]
    pub gradient_hsv: Option<bool>,
    /// Colour stops sampled by depth in palette modes. After sanitizing, fewer than 2 stops are filled from the start and end colours. Each stop is RGB, channels 0..=255.
    #[serde(default)]
    pub palette: Option<Vec<[u8; 3]>>,
    /// Seed for `random` colour mode. Any u64.
    #[serde(default)]
    pub color_seed: Option<u64>,
    /// Canvas colour for solid backgrounds, and the first colour of linear and radial backgrounds. RGB, channels 0..=255.
    #[serde(default)]
    pub background: Option<[u8; 3]>,
    /// Canvas fill: `solid`, `linear`, or `radial`. Solid uses only `background`.
    #[serde(default)]
    pub bg_mode: Option<BgModeDto>,
    /// Second background colour for `linear` and `radial`. RGB, channels 0..=255.
    #[serde(default)]
    pub bg_end: Option<[u8; 3]>,
    /// Linear gradient direction in degrees. 0 runs left to right; screen y grows downward. Non-finite values are reset to 0.
    #[serde(default)]
    pub bg_angle: Option<f32>,
    /// Stroke width as a fraction of the mean target side length. Clamped to 0.001..=0.2.
    #[serde(default)]
    pub stroke_fraction: Option<f32>,
    /// Fillet size. Clamped to 0..=1. 0 keeps sharp corners; 1 is the largest fillet that leaves each edge midpoint straight.
    #[serde(default)]
    pub corner_radius: Option<f32>,
    /// Vertex offset as a fraction of the longest seed edge, scaled with each instance. Clamped to 0..=0.5. 0 leaves vertices unchanged.
    #[serde(default)]
    pub jitter: Option<f32>,
    /// Chooses the deterministic offsets used when `jitter` is above 0. Any u64.
    #[serde(default)]
    pub jitter_seed: Option<u64>,
    /// Stroke coverage. Clamped to 0..=1. 0 is invisible, 1 is fully opaque.
    #[serde(default)]
    pub stroke_opacity: Option<f32>,
    /// How much deeper layers fade. Clamped to 0..=1. 0 keeps every depth at `stroke_opacity`.
    #[serde(default)]
    pub opacity_fade: Option<f32>,
    /// How much stroke width shrinks toward deeper layers. Clamped to 0..=1. 0 keeps every depth at full width.
    #[serde(default)]
    pub stroke_taper: Option<f32>,
    /// Interior fill: `none` (strokes only), `solid` (`color_start`), or `palette`.
    #[serde(default)]
    pub fill_mode: Option<FillModeDto>,
    /// Fill coverage. Clamped to 0..=1.
    #[serde(default)]
    pub fill_opacity: Option<f32>,
    /// Rotational copies about the origin. Clamped to 1..=24. 1 leaves a single copy.
    #[serde(default)]
    pub symmetry: Option<u32>,
    /// When true, add a reflection of each sector across the x-axis.
    #[serde(default)]
    pub mirror: Option<bool>,
    /// Bloom strength. Clamped to 0..=4. 0 skips the post-process.
    #[serde(default)]
    pub glow: Option<f32>,
    /// Bloom blur radius in pixels. Clamped to 0..=64.
    #[serde(default)]
    pub glow_radius: Option<f32>,
}

impl ShapeParamsDto {
    /// Defaults, then caller overrides, then the engine's sanitizer.
    pub fn into_params(self) -> FractalParams {
        let mut params = FractalParams::default();
        if let Some(side_lengths) = self.side_lengths {
            params.side_lengths = side_lengths;
        }
        if let Some(shrink) = self.shrink {
            params.shrink = shrink;
        }
        if let Some(shrink_by_depth) = self.shrink_by_depth {
            params.shrink_by_depth = shrink_by_depth;
        }
        if let Some(depth) = self.depth {
            params.depth = depth;
        }
        if let Some(rotation_deg) = self.rotation_deg {
            params.rotation_deg = rotation_deg;
        }
        if let Some(inward) = self.inward {
            params.inward = inward;
        }
        if let Some(outward_only) = self.outward_only {
            params.outward_only = outward_only;
        }
        if let Some(twist_deg) = self.twist_deg {
            params.twist_deg = twist_deg;
        }
        if let Some(colour_mode) = self.colour_mode {
            params.colour_mode = colour_mode.into();
        }
        if let Some(color_start) = self.color_start {
            params.color_start = color_start;
        }
        if let Some(color_end) = self.color_end {
            params.color_end = color_end;
        }
        if let Some(gradient_hsv) = self.gradient_hsv {
            params.gradient_hsv = gradient_hsv;
        }
        if let Some(palette) = self.palette {
            params.palette = palette;
        }
        if let Some(color_seed) = self.color_seed {
            params.color_seed = color_seed;
        }
        if let Some(background) = self.background {
            params.background = background;
        }
        if let Some(bg_mode) = self.bg_mode {
            params.bg_mode = bg_mode.into();
        }
        if let Some(bg_end) = self.bg_end {
            params.bg_end = bg_end;
        }
        if let Some(bg_angle) = self.bg_angle {
            params.bg_angle = bg_angle;
        }
        if let Some(stroke_fraction) = self.stroke_fraction {
            params.stroke_fraction = stroke_fraction;
        }
        if let Some(corner_radius) = self.corner_radius {
            params.corner_radius = corner_radius;
        }
        if let Some(jitter) = self.jitter {
            params.jitter = jitter;
        }
        if let Some(jitter_seed) = self.jitter_seed {
            params.jitter_seed = jitter_seed;
        }
        if let Some(stroke_opacity) = self.stroke_opacity {
            params.stroke_opacity = stroke_opacity;
        }
        if let Some(opacity_fade) = self.opacity_fade {
            params.opacity_fade = opacity_fade;
        }
        if let Some(stroke_taper) = self.stroke_taper {
            params.stroke_taper = stroke_taper;
        }
        if let Some(fill_mode) = self.fill_mode {
            params.fill_mode = fill_mode.into();
        }
        if let Some(fill_opacity) = self.fill_opacity {
            params.fill_opacity = fill_opacity;
        }
        if let Some(symmetry) = self.symmetry {
            params.symmetry = symmetry;
        }
        if let Some(mirror) = self.mirror {
            params.mirror = mirror;
        }
        if let Some(glow) = self.glow {
            params.glow = glow;
        }
        if let Some(glow_radius) = self.glow_radius {
            params.glow_radius = glow_radius;
        }
        params.sanitized()
    }
}

/// Shape parameters after defaults and clamping, echoed back to the caller.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedShapeParams {
    pub side_lengths: Vec<f32>,
    pub shrink: f32,
    pub shrink_by_depth: Vec<f32>,
    pub depth: u32,
    pub rotation_deg: f32,
    pub inward: bool,
    pub outward_only: bool,
    pub twist_deg: f32,
    pub colour_mode: ColourModeDto,
    pub color_start: [u8; 3],
    pub color_end: [u8; 3],
    pub gradient_hsv: bool,
    pub palette: Vec<[u8; 3]>,
    pub color_seed: u64,
    pub background: [u8; 3],
    pub bg_mode: BgModeDto,
    pub bg_end: [u8; 3],
    pub bg_angle: f32,
    pub stroke_fraction: f32,
    pub corner_radius: f32,
    pub jitter: f32,
    pub jitter_seed: u64,
    pub stroke_opacity: f32,
    pub opacity_fade: f32,
    pub stroke_taper: f32,
    pub fill_mode: FillModeDto,
    pub fill_opacity: f32,
    pub symmetry: u32,
    pub mirror: bool,
    pub glow: f32,
    pub glow_radius: f32,
}

impl From<&FractalParams> for ResolvedShapeParams {
    fn from(params: &FractalParams) -> Self {
        Self {
            side_lengths: params.side_lengths.clone(),
            shrink: params.shrink,
            shrink_by_depth: params.shrink_by_depth.clone(),
            depth: params.depth,
            rotation_deg: params.rotation_deg,
            inward: params.inward,
            outward_only: params.outward_only,
            twist_deg: params.twist_deg,
            colour_mode: params.colour_mode.into(),
            color_start: params.color_start,
            color_end: params.color_end,
            gradient_hsv: params.gradient_hsv,
            palette: params.palette.clone(),
            color_seed: params.color_seed,
            background: params.background,
            bg_mode: params.bg_mode.into(),
            bg_end: params.bg_end,
            bg_angle: params.bg_angle,
            stroke_fraction: params.stroke_fraction,
            corner_radius: params.corner_radius,
            jitter: params.jitter,
            jitter_seed: params.jitter_seed,
            stroke_opacity: params.stroke_opacity,
            opacity_fade: params.opacity_fade,
            stroke_taper: params.stroke_taper,
            fill_mode: params.fill_mode.into(),
            fill_opacity: params.fill_opacity,
            symmetry: params.symmetry,
            mirror: params.mirror,
            glow: params.glow,
            glow_radius: params.glow_radius,
        }
    }
}

/// One stamp definition. `params` omitted means a default shape fractal.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MotifInput {
    /// Display name. Empty or omitted becomes "Motif {n}".
    #[serde(default)]
    pub name: Option<String>,
    /// Shape parameters for this stamp. Omitted fields use the engine defaults, then clamping.
    #[serde(default)]
    pub params: ShapeParamsDto,
}

/// Placement settings for stamping motifs. Omitted fields keep the engine defaults, then every value is clamped.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ArrangementParamsDto {
    /// Seed polygon side lengths. The count is clamped to 3..=16. Each length is clamped to 0.05..=8.
    #[serde(default)]
    pub side_lengths: Option<Vec<f32>>,
    /// Child size as a fraction of its parent. Clamped to 0.05..=0.95.
    #[serde(default)]
    pub shrink: Option<f32>,
    /// Per-depth shrink factors, each clamped to 0.05..=0.95. Empty or omitted uses `shrink` at every depth.
    #[serde(default)]
    pub shrink_by_depth: Option<Vec<f32>>,
    /// Spacing between a stamp and the next layer. Clamped to -4..=4. 0 keeps the current spacing, negative values sit toward the parent, and positive values spread the layers apart.
    #[serde(default)]
    pub gap: Option<f32>,
    /// Recursion depth. Clamped to 0..=16. Level 0 is the root stamp, so there are `depth + 1` motif slots.
    #[serde(default)]
    pub depth: Option<u32>,
    /// Root rotation in degrees. Non-finite values are reset to 0.
    #[serde(default)]
    pub rotation_deg: Option<f32>,
    /// When true, children grow toward the parent centroid.
    #[serde(default)]
    pub inward: Option<bool>,
    /// When true, skip a stamp whose body points back toward the origin. The root stamp is always placed.
    #[serde(default)]
    pub outward_only: Option<bool>,
    /// Extra rotation applied to each child, in degrees. Non-finite values are reset to 0.
    #[serde(default)]
    pub twist_deg: Option<f32>,
    /// Canvas colour for solid backgrounds, and the first colour of linear and radial backgrounds. RGB, channels 0..=255.
    #[serde(default)]
    pub background: Option<[u8; 3]>,
    /// Canvas fill: `solid`, `linear`, or `radial`.
    #[serde(default)]
    pub bg_mode: Option<BgModeDto>,
    /// Second background colour for `linear` and `radial`. RGB, channels 0..=255.
    #[serde(default)]
    pub bg_end: Option<[u8; 3]>,
    /// Linear gradient direction in degrees. 0 runs left to right. Non-finite values are reset to 0.
    #[serde(default)]
    pub bg_angle: Option<f32>,
    /// Rotational copies about the origin. Clamped to 1..=24.
    #[serde(default)]
    pub symmetry: Option<u32>,
    /// When true, add a reflection of each sector across the x-axis.
    #[serde(default)]
    pub mirror: Option<bool>,
    /// Bloom strength. Clamped to 0..=4. 0 skips the post-process.
    #[serde(default)]
    pub glow: Option<f32>,
    /// Bloom blur radius in pixels. Clamped to 0..=64.
    #[serde(default)]
    pub glow_radius: Option<f32>,
}

impl ArrangementParamsDto {
    /// Defaults, then caller overrides. `motif_by_depth` holds engine motif ids (not indices).
    pub fn into_params(self, motif_by_depth: Vec<u64>) -> ArrangementParams {
        let mut params = ArrangementParams::default();
        if let Some(side_lengths) = self.side_lengths {
            params.side_lengths = side_lengths;
        }
        if let Some(shrink) = self.shrink {
            params.shrink = shrink;
        }
        if let Some(shrink_by_depth) = self.shrink_by_depth {
            params.shrink_by_depth = shrink_by_depth;
        }
        if let Some(gap) = self.gap {
            params.gap = gap;
        }
        if let Some(depth) = self.depth {
            params.depth = depth;
        }
        if let Some(rotation_deg) = self.rotation_deg {
            params.rotation_deg = rotation_deg;
        }
        if let Some(inward) = self.inward {
            params.inward = inward;
        }
        if let Some(outward_only) = self.outward_only {
            params.outward_only = outward_only;
        }
        if let Some(twist_deg) = self.twist_deg {
            params.twist_deg = twist_deg;
        }
        if let Some(background) = self.background {
            params.background = background;
        }
        if let Some(bg_mode) = self.bg_mode {
            params.bg_mode = bg_mode.into();
        }
        if let Some(bg_end) = self.bg_end {
            params.bg_end = bg_end;
        }
        if let Some(bg_angle) = self.bg_angle {
            params.bg_angle = bg_angle;
        }
        if let Some(symmetry) = self.symmetry {
            params.symmetry = symmetry;
        }
        if let Some(mirror) = self.mirror {
            params.mirror = mirror;
        }
        if let Some(glow) = self.glow {
            params.glow = glow;
        }
        if let Some(glow_radius) = self.glow_radius {
            params.glow_radius = glow_radius;
        }
        if !motif_by_depth.is_empty() {
            params.motif_by_depth = motif_by_depth;
        }
        params.sanitized()
    }
}

/// Arrangement parameters after defaults and clamping.
///
/// `motif_by_depth` is indices into the request's `motifs` list, after the
/// engine pads a short list and drops entries past `depth`.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedArrangementParams {
    pub side_lengths: Vec<f32>,
    pub shrink: f32,
    pub shrink_by_depth: Vec<f32>,
    pub gap: f32,
    pub depth: u32,
    pub rotation_deg: f32,
    pub inward: bool,
    pub outward_only: bool,
    pub twist_deg: f32,
    pub background: [u8; 3],
    pub bg_mode: BgModeDto,
    pub bg_end: [u8; 3],
    pub bg_angle: f32,
    pub motif_by_depth: Vec<usize>,
    pub symmetry: u32,
    pub mirror: bool,
    pub glow: f32,
    pub glow_radius: f32,
}

impl From<&ArrangementParams> for ResolvedArrangementParams {
    fn from(params: &ArrangementParams) -> Self {
        Self {
            side_lengths: params.side_lengths.clone(),
            shrink: params.shrink,
            shrink_by_depth: params.shrink_by_depth.clone(),
            gap: params.gap,
            depth: params.depth,
            rotation_deg: params.rotation_deg,
            inward: params.inward,
            outward_only: params.outward_only,
            twist_deg: params.twist_deg,
            background: params.background,
            bg_mode: params.bg_mode.into(),
            bg_end: params.bg_end,
            bg_angle: params.bg_angle,
            motif_by_depth: params
                .motif_by_depth
                .iter()
                .map(|id| id.saturating_sub(1) as usize)
                .collect(),
            symmetry: params.symmetry,
            mirror: params.mirror,
            glow: params.glow,
            glow_radius: params.glow_radius,
        }
    }
}

/// Map `motif_by_depth` indices to engine ids (`index + 1`).
///
/// An empty list selects motif 0 for every depth. [`ArrangementParams::sanitized`]
/// then repeats the last id until there is one slot per level `0..=depth`.
pub fn motif_ids(indices: &[usize], motif_count: usize) -> Result<Vec<u64>, String> {
    if motif_count == 0 {
        return Err("at least one motif is required".to_string());
    }
    if indices.is_empty() {
        return Ok(vec![1]);
    }
    let mut ids = Vec::with_capacity(indices.len());
    for (level, index) in indices.iter().copied().enumerate() {
        if index >= motif_count {
            return Err(format!(
                "motif_by_depth[{level}] is {index}, but motifs has length {motif_count} (valid indices are 0..{motif_count})"
            ));
        }
        ids.push(index as u64 + 1);
    }
    Ok(ids)
}

/// Render a shape fractal. Shape fields are flattened beside the file options.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RenderShapeRequest {
    /// Shape parameters. Every field is optional.
    #[serde(flatten)]
    pub shape: ShapeParamsDto,
    /// Image width in pixels. Clamped to 16..=16384.
    pub width: u32,
    /// Image height in pixels. Clamped to 16..=16384.
    pub height: u32,
    /// `png` or `svg`.
    pub format: ImageFormat,
    /// File path to write. Parent directories are created. Relative paths use the server process working directory.
    pub output_path: String,
    /// When true, also return the written file inline. PNG is an image content block; SVG is text. Defaults to false.
    #[serde(default)]
    pub return_image: bool,
}

/// Stamp motifs with the arrangement growth rule. PNG only.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RenderArrangementRequest {
    /// Arrangement parameters. Every field is optional.
    #[serde(flatten)]
    pub arrangement: ArrangementParamsDto,
    /// Stamps to place. Each entry's `params` is a full shape option set. Motif ids are assigned as index + 1.
    pub motifs: Vec<MotifInput>,
    /// Indices into `motifs`, one per depth level from 0 through `depth`. Omitted or empty uses motif 0 for every level. Out-of-range indices are rejected. A short list repeats its last index; extra entries are validated and then dropped to `depth + 1` slots.
    #[serde(default)]
    pub motif_by_depth: Vec<usize>,
    /// Image width in pixels. Clamped to 16..=16384.
    pub width: u32,
    /// Image height in pixels. Clamped to 16..=16384.
    pub height: u32,
    /// PNG path to write. SVG is rejected. Parent directories are created.
    pub output_path: String,
    /// When true, also return the PNG inline. Defaults to false.
    #[serde(default)]
    pub return_image: bool,
    /// Optional format. Only `png` is accepted. Omitted means PNG. `svg` is rejected because the engine writes SVG for shape fractals only.
    #[serde(default)]
    pub format: Option<ImageFormat>,
}

/// Pick a random valid shape with `FractalParams::randomize` and render it.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RandomizeShapeRequest {
    /// Seed passed to `FractalParams::randomize`. The same seed repeats the same parameters.
    pub seed: u64,
    /// Image width in pixels. Clamped to 16..=16384.
    pub width: u32,
    /// Image height in pixels. Clamped to 16..=16384.
    pub height: u32,
    /// `png` or `svg`.
    pub format: ImageFormat,
    /// File path to write. Parent directories are created.
    pub output_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use polygon_fractal::geom::{BgMode, ColourMode, FillMode, MAX_DEPTH};

    #[test]
    fn omitted_shape_fields_match_sanitized_defaults() {
        let params = ShapeParamsDto::default().into_params();
        assert_eq!(params, FractalParams::default().sanitized());
    }

    #[test]
    fn shape_overrides_are_clamped_by_the_engine() {
        let dto: ShapeParamsDto = serde_json::from_str(
            r#"{
                "colour_mode": "palette",
                "bg_mode": "radial",
                "fill_mode": "solid",
                "depth": 100,
                "shrink": 0.01,
                "symmetry": 0,
                "glow": 9.0,
                "side_lengths": [1.0, 2.0]
            }"#,
        )
        .unwrap();
        let params = dto.into_params();
        assert_eq!(params.colour_mode, ColourMode::Palette);
        assert_eq!(params.bg_mode, BgMode::Radial);
        assert_eq!(params.fill_mode, FillMode::Solid);
        assert_eq!(params.depth, MAX_DEPTH);
        assert_eq!(params.shrink, 0.05);
        assert_eq!(params.symmetry, 1);
        assert_eq!(params.glow, 4.0);
        assert_eq!(params.side_lengths.len(), 3);
    }

    #[test]
    fn arrangement_gap_and_depth_are_clamped() {
        let dto: ArrangementParamsDto =
            serde_json::from_str(r#"{"gap": 99.0, "depth": 40, "bg_mode": "linear"}"#).unwrap();
        let params = dto.into_params(vec![1]);
        assert_eq!(params.gap, 4.0);
        assert_eq!(params.depth, MAX_DEPTH);
        assert_eq!(params.bg_mode, BgMode::Linear);
        assert_eq!(params.motif_by_depth.len(), MAX_DEPTH as usize + 1);
        assert!(params.motif_by_depth.iter().all(|id| *id == 1));
    }

    #[test]
    fn outward_only_is_a_tool_field_for_shape_and_arrangement() {
        let shape: ShapeParamsDto = serde_json::from_str(r#"{"outward_only": true}"#).unwrap();
        let shape_params = shape.into_params();
        assert!(shape_params.outward_only);
        assert!(ResolvedShapeParams::from(&shape_params).outward_only);
        let shape_off: ShapeParamsDto = serde_json::from_str(r#"{"outward_only": false}"#).unwrap();
        let shape_off_params = shape_off.into_params();
        assert!(!shape_off_params.outward_only);
        assert!(!ResolvedShapeParams::from(&shape_off_params).outward_only);
        assert!(!ShapeParamsDto::default().into_params().outward_only);

        let arrangement: ArrangementParamsDto =
            serde_json::from_str(r#"{"outward_only": true}"#).unwrap();
        let params = arrangement.into_params(vec![1]);
        assert!(params.outward_only);
        assert!(ResolvedArrangementParams::from(&params).outward_only);

        let shape_schema = serde_json::to_value(schemars::schema_for!(RenderShapeRequest)).unwrap();
        let arrangement_schema =
            serde_json::to_value(schemars::schema_for!(RenderArrangementRequest)).unwrap();
        assert!(shape_schema["properties"]["outward_only"].is_object());
        assert!(arrangement_schema["properties"]["outward_only"].is_object());
        assert!(shape_schema["properties"]["outward_only"]["description"]
            .as_str()
            .unwrap()
            .contains("origin"));
    }

    #[test]
    fn motif_indices_must_be_in_range() {
        assert_eq!(motif_ids(&[], 2).unwrap(), vec![1]);
        assert_eq!(motif_ids(&[0, 1], 2).unwrap(), vec![1, 2]);
        assert!(motif_ids(&[2], 2).is_err());
        assert!(motif_ids(&[], 0).is_err());
    }
}

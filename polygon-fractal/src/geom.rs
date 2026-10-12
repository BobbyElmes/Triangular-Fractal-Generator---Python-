//! Closed-polygon edge fractal.
//!
//! The seed is a polygon whose side lengths are walked with equal exterior
//! turns, then closed by spreading the leftover gap around the outline.
//! Each free edge grows a uniformly scaled copy of that seed.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Hard stop so a high side count and depth cannot allocate without bound.
const MAX_SEGMENTS: usize = 1_500_000;
/// Hard stop so an arrangement cannot blit without bound.
const MAX_INSTANCES: usize = 4_096;
/// Gap slider ends. At `-GAP_LIMIT` each layer sits on its parent.
pub const GAP_LIMIT: f32 = 4.0;

const MIN_SIDE: f32 = 0.05;
const MARGIN: f32 = 0.04;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn centroid(verts: &[Vec2]) -> Vec2 {
    let n = verts.len() as f32;
    let sum = verts.iter().copied().fold(Vec2::ZERO, |acc, v| acc + v);
    sum * (1.0 / n)
}

fn rotate_around(point: Vec2, pivot: Vec2, radians: f32) -> Vec2 {
    let (sin, cos) = radians.sin_cos();
    let d = point - pivot;
    pivot + Vec2::new(d.x * cos - d.y * sin, d.x * sin + d.y * cos)
}

fn reflect_across(point: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 < 1e-12 {
        return point;
    }
    let t = (point - a).dot(ab) / len2;
    let proj = a + ab * t;
    proj * 2.0 - point
}

/// How the canvas behind the figure is painted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgMode {
    /// Flat `background` colour.
    Solid,
    /// Blend `background` into `bg_end` along `bg_angle`.
    Linear,
    /// Blend `background` at the centre into `bg_end` at the corners.
    Radial,
}

/// How each outline is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourMode {
    /// Every line uses `color_start`.
    Single,
    /// Blend `color_start` into `color_end` by depth.
    Gradient,
    /// A fresh colour for each line, from `color_seed`.
    Random,
    /// Sample `palette` by depth, interpolating between adjacent stops.
    Palette,
}

/// How each polygon instance is filled. Strokes are always drawn afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMode {
    /// Strokes only. Outline rings are not built.
    None,
    /// Every instance uses `color_start`.
    Solid,
    /// Each instance uses the palette sample at its depth.
    Palette,
}

/// Parameters edited by the control panel.
#[derive(Clone, Debug, PartialEq)]
pub struct FractalParams {
    pub side_lengths: Vec<f32>,
    pub shrink: f32,
    /// Empty uses `shrink` at every depth. Entry `d` is the factor for a child created at depth `d`.
    pub shrink_by_depth: Vec<f32>,
    pub depth: u32,
    pub rotation_deg: f32,
    pub inward: bool,
    pub twist_deg: f32,
    pub colour_mode: ColourMode,
    pub color_start: [u8; 3],
    pub color_end: [u8; 3],
    /// When true, gradient and palette modes blend between colours in HSV (shortest hue).
    /// False keeps the per-channel RGB blend.
    pub gradient_hsv: bool,
    /// Stops sampled by depth when `colour_mode` is [`ColourMode::Palette`].
    pub palette: Vec<[u8; 3]>,
    /// Seed for [`ColourMode::Random`]. A new value gives a new set of colours.
    pub color_seed: u64,
    pub background: [u8; 3],
    /// Canvas fill. [`BgMode::Solid`] uses only `background`.
    pub bg_mode: BgMode,
    /// Second colour for [`BgMode::Linear`] and [`BgMode::Radial`].
    pub bg_end: [u8; 3],
    /// Linear gradient direction in degrees. 0 runs left to right; screen y grows downward.
    pub bg_angle: f32,
    /// Stroke width as a fraction of the mean target side length.
    pub stroke_fraction: f32,
    /// Fillet size from 0 (sharp corners) to 1 (largest fillet that leaves each edge midpoint straight).
    pub corner_radius: f32,
    /// Vertex offset as a fraction of the longest seed edge, scaled with each instance. 0 leaves vertices unchanged.
    pub jitter: f32,
    /// Chooses the deterministic offsets used when `jitter` is above 0.
    pub jitter_seed: u64,
    /// Stroke coverage from 0 (invisible) to 1 (fully opaque).
    pub stroke_opacity: f32,
    /// How much deeper layers fade. 0 keeps every depth at `stroke_opacity`.
    pub opacity_fade: f32,
    /// How much stroke width shrinks toward deeper layers. 0 keeps every depth at full width.
    pub stroke_taper: f32,
    /// Interior fill. [`FillMode::None`] keeps the stroke-only figure.
    pub fill_mode: FillMode,
    /// Fill coverage from 0 (invisible) to 1 (fully opaque).
    pub fill_opacity: f32,
    /// Rotational copies about the origin. 1 leaves a single copy.
    pub symmetry: u32,
    /// Add a reflection of each sector across the x-axis.
    pub mirror: bool,
    /// Bloom strength. 0 skips the post-process and leaves pixels unchanged.
    pub glow: f32,
    /// Bloom blur radius in pixels, clamped to 64.
    pub glow_radius: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            side_lengths: vec![1.0, 1.0, 1.0],
            shrink: 0.5,
            shrink_by_depth: Vec::new(),
            depth: 6,
            rotation_deg: 0.0,
            inward: false,
            twist_deg: 0.0,
            colour_mode: ColourMode::Gradient,
            color_start: [255, 200, 0],
            color_end: [165, 242, 243],
            gradient_hsv: false,
            palette: vec![[255, 200, 0], [210, 221, 122], [165, 242, 243]],
            color_seed: 1,
            background: [0, 0, 0],
            bg_mode: BgMode::Solid,
            bg_end: [48, 72, 120],
            bg_angle: 0.0,
            stroke_fraction: 0.02,
            corner_radius: 0.0,
            jitter: 0.0,
            jitter_seed: 1,
            stroke_opacity: 1.0,
            opacity_fade: 0.0,
            stroke_taper: 0.0,
            fill_mode: FillMode::None,
            fill_opacity: 1.0,
            symmetry: 1,
            mirror: false,
            glow: 0.0,
            glow_radius: 8.0,
        }
    }
}

impl FractalParams {
    pub fn sanitized(&self) -> Self {
        let mut params = self.clone();
        let n = params.side_lengths.len().clamp(3, 16);
        set_side_count(&mut params.side_lengths, n);
        for length in &mut params.side_lengths {
            if !length.is_finite() {
                *length = 1.0;
            }
            *length = length.clamp(MIN_SIDE, 8.0);
        }
        if !params.shrink.is_finite() {
            params.shrink = 0.5;
        }
        params.shrink = params.shrink.clamp(0.05, 0.95);
        for value in &mut params.shrink_by_depth {
            if !value.is_finite() {
                *value = params.shrink;
            }
            *value = value.clamp(0.05, 0.95);
        }
        params.depth = params.depth.clamp(1, 12);
        if !params.rotation_deg.is_finite() {
            params.rotation_deg = 0.0;
        }
        if !params.twist_deg.is_finite() {
            params.twist_deg = 0.0;
        }
        if !params.stroke_fraction.is_finite() {
            params.stroke_fraction = 0.02;
        }
        params.stroke_fraction = params.stroke_fraction.clamp(0.001, 0.2);
        if !params.corner_radius.is_finite() {
            params.corner_radius = 0.0;
        }
        params.corner_radius = params.corner_radius.clamp(0.0, 1.0);
        if !params.jitter.is_finite() {
            params.jitter = 0.0;
        }
        params.jitter = params.jitter.clamp(0.0, 0.5);
        if !params.stroke_opacity.is_finite() {
            params.stroke_opacity = 1.0;
        }
        params.stroke_opacity = params.stroke_opacity.clamp(0.0, 1.0);
        if !params.opacity_fade.is_finite() {
            params.opacity_fade = 0.0;
        }
        params.opacity_fade = params.opacity_fade.clamp(0.0, 1.0);
        if !params.stroke_taper.is_finite() {
            params.stroke_taper = 0.0;
        }
        params.stroke_taper = params.stroke_taper.clamp(0.0, 1.0);
        if !params.fill_opacity.is_finite() {
            params.fill_opacity = 1.0;
        }
        params.fill_opacity = params.fill_opacity.clamp(0.0, 1.0);
        params.symmetry = params.symmetry.clamp(1, 24);
        if !params.bg_angle.is_finite() {
            params.bg_angle = 0.0;
        }
        if !params.glow.is_finite() {
            params.glow = 0.0;
        }
        params.glow = params.glow.clamp(0.0, 4.0);
        if !params.glow_radius.is_finite() {
            params.glow_radius = 8.0;
        }
        params.glow_radius = params.glow_radius.clamp(0.0, 64.0);
        match params.palette.len() {
            0 => params.palette = vec![params.color_start, params.color_end],
            1 => params.palette.push(params.color_end),
            _ => {}
        }
        params
    }

    /// Replace the editable fields with a random valid setup. `seed` selects the result.
    pub fn randomize(&mut self, seed: u64) {
        let mut rng = ColourRng(seed);
        let sides = rng.gen_range_u32(3, 16) as usize;
        self.side_lengths = (0..sides).map(|_| rng.gen_range_f32(MIN_SIDE, 8.0)).collect();
        self.shrink = rng.gen_range_f32(0.05, 0.95);
        self.shrink_by_depth.clear();
        self.depth = rng.gen_range_u32(1, 12);
        self.rotation_deg = rng.gen_range_f32(-180.0, 180.0);
        self.twist_deg = rng.gen_range_f32(-180.0, 180.0);
        self.inward = rng.gen_bool();
        self.colour_mode = match rng.gen_range_u32(0, 3) {
            0 => ColourMode::Single,
            1 => ColourMode::Gradient,
            2 => ColourMode::Random,
            _ => ColourMode::Palette,
        };
        self.color_start = rng.next_colour();
        self.color_end = rng.next_colour();
        self.gradient_hsv = rng.gen_bool();
        let stops = rng.gen_range_u32(2, 6) as usize;
        self.palette = (0..stops).map(|_| rng.next_colour()).collect();
        self.color_seed = rng.next_u64();
        self.stroke_fraction = rng.gen_range_f32(0.001, 0.2);
        self.corner_radius = rng.gen_range_f32(0.0, 1.0);
        self.jitter = rng.gen_range_f32(0.0, 0.5);
        self.jitter_seed = rng.next_u64();
        // Keep strokes visible; a fully transparent surprise reads as a blank canvas.
        self.stroke_opacity = rng.gen_range_f32(0.25, 1.0);
        self.opacity_fade = rng.gen_range_f32(0.0, 1.0);
        self.stroke_taper = rng.gen_range_f32(0.0, 1.0);
        *self = self.sanitized();
    }
}

/// A saved fractal image that an arrangement can stamp.
#[derive(Clone, Debug)]
pub struct Motif {
    pub id: u64,
    pub name: String,
    pub revision: u64,
    pub params: FractalParams,
}

/// Placement settings for stamping motifs with the same edge-growth rule.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrangementParams {
    pub side_lengths: Vec<f32>,
    pub shrink: f32,
    /// Empty uses `shrink` at every depth. Entry `d` is the factor for a child created at depth `d`.
    pub shrink_by_depth: Vec<f32>,
    /// Spacing between a stamp and the next layer.
    /// 0 keeps the current spacing, the negative end sits on the parent, and positive values spread the layers apart.
    pub gap: f32,
    pub depth: u32,
    pub rotation_deg: f32,
    pub inward: bool,
    pub twist_deg: f32,
    pub background: [u8; 3],
    /// Canvas fill. [`BgMode::Solid`] uses only `background`.
    pub bg_mode: BgMode,
    /// Second colour for [`BgMode::Linear`] and [`BgMode::Radial`].
    pub bg_end: [u8; 3],
    /// Linear gradient direction in degrees. 0 runs left to right; screen y grows downward.
    pub bg_angle: f32,
    /// Motif id for each depth. Index 0 is the root stamp.
    pub motif_by_depth: Vec<u64>,
    /// Rotational copies about the origin. 1 leaves a single copy.
    pub symmetry: u32,
    /// Add a reflection of each sector across the x-axis.
    pub mirror: bool,
    /// Bloom strength. 0 skips the post-process and leaves pixels unchanged.
    pub glow: f32,
    /// Bloom blur radius in pixels, clamped to 64.
    pub glow_radius: f32,
}

impl Default for ArrangementParams {
    fn default() -> Self {
        Self {
            side_lengths: vec![1.0, 1.0, 1.0],
            shrink: 0.5,
            shrink_by_depth: Vec::new(),
            gap: 0.0,
            depth: 2,
            rotation_deg: 0.0,
            inward: false,
            twist_deg: 0.0,
            background: [0, 0, 0],
            bg_mode: BgMode::Solid,
            bg_end: [48, 72, 120],
            bg_angle: 0.0,
            motif_by_depth: vec![1, 1, 1],
            symmetry: 1,
            mirror: false,
            glow: 0.0,
            glow_radius: 8.0,
        }
    }
}

impl ArrangementParams {
    pub fn sanitized(&self) -> Self {
        let mut params = self.clone();
        let n = params.side_lengths.len().clamp(3, 16);
        set_side_count(&mut params.side_lengths, n);
        for length in &mut params.side_lengths {
            if !length.is_finite() {
                *length = 1.0;
            }
            *length = length.clamp(MIN_SIDE, 8.0);
        }
        if !params.shrink.is_finite() {
            params.shrink = 0.5;
        }
        params.shrink = params.shrink.clamp(0.05, 0.95);
        for value in &mut params.shrink_by_depth {
            if !value.is_finite() {
                *value = params.shrink;
            }
            *value = value.clamp(0.05, 0.95);
        }
        if !params.gap.is_finite() {
            params.gap = 0.0;
        }
        params.gap = params.gap.clamp(-GAP_LIMIT, GAP_LIMIT);
        params.depth = params.depth.clamp(0, 12);
        if !params.rotation_deg.is_finite() {
            params.rotation_deg = 0.0;
        }
        if !params.twist_deg.is_finite() {
            params.twist_deg = 0.0;
        }
        if !params.bg_angle.is_finite() {
            params.bg_angle = 0.0;
        }
        params.symmetry = params.symmetry.clamp(1, 24);
        if !params.glow.is_finite() {
            params.glow = 0.0;
        }
        params.glow = params.glow.clamp(0.0, 4.0);
        if !params.glow_radius.is_finite() {
            params.glow_radius = 8.0;
        }
        params.glow_radius = params.glow_radius.clamp(0.0, 64.0);
        if params.motif_by_depth.is_empty() {
            params.motif_by_depth.push(0);
        }
        let slots = params.depth as usize + 1;
        if params.motif_by_depth.len() < slots {
            let fill = *params.motif_by_depth.last().unwrap();
            params.motif_by_depth.resize(slots, fill);
        } else {
            params.motif_by_depth.truncate(slots);
        }
        params
    }

    /// Replace placement fields with a random valid setup. Motif choices are kept.
    pub fn randomize(&mut self, seed: u64) {
        let mut rng = ColourRng(seed);
        let sides = rng.gen_range_u32(3, 16) as usize;
        self.side_lengths = (0..sides).map(|_| rng.gen_range_f32(MIN_SIDE, 8.0)).collect();
        self.shrink = rng.gen_range_f32(0.05, 0.95);
        self.shrink_by_depth.clear();
        self.gap = rng.gen_range_f32(-GAP_LIMIT, GAP_LIMIT);
        self.depth = rng.gen_range_u32(0, 12);
        self.rotation_deg = rng.gen_range_f32(-180.0, 180.0);
        self.twist_deg = rng.gen_range_f32(-180.0, 180.0);
        self.inward = rng.gen_bool();
        *self = self.sanitized();
    }
}

pub fn set_side_count(lengths: &mut Vec<f32>, n: usize) {
    let n = n.clamp(3, 16);
    let average = if lengths.is_empty() {
        1.0
    } else {
        lengths.iter().sum::<f32>() / lengths.len() as f32
    };
    if lengths.len() < n {
        lengths.resize(n, average);
    } else {
        lengths.truncate(n);
    }
}

pub fn equalize(lengths: &mut [f32]) {
    if lengths.is_empty() {
        return;
    }
    let average = lengths.iter().sum::<f32>() / lengths.len() as f32;
    for length in lengths {
        *length = average;
    }
}

#[derive(Clone, Debug)]
pub struct SeedShape {
    pub vertices: Vec<Vec2>,
    /// Gap length divided by the perimeter of the target sides, before closure.
    pub gap_ratio: f32,
    pub mean_target: f32,
    pub max_edge: f32,
}

/// Walk equal exterior turns, then spread the gap so the outline meets.
pub fn close_seed(lengths: &[f32]) -> SeedShape {
    let n = lengths.len();
    debug_assert!(n >= 3);
    let turn = std::f32::consts::TAU / n as f32;
    let mut heading = 0.0_f32;
    let mut points = Vec::with_capacity(n + 1);
    let mut cursor = Vec2::ZERO;
    points.push(cursor);
    for &len in lengths {
        cursor.x += len * heading.cos();
        cursor.y += len * heading.sin();
        points.push(cursor);
        heading += turn;
    }

    let gap = points[n] - points[0];
    let perimeter = lengths.iter().sum::<f32>().max(1e-6);
    let gap_ratio = gap.length() / perimeter;
    for k in 0..=n {
        let t = k as f32 / n as f32;
        points[k] = points[k] - gap * t;
    }
    points.pop();

    let center = centroid(&points);
    for point in &mut points {
        *point = *point - center;
    }

    let max_edge = points
        .iter()
        .enumerate()
        .map(|(i, v)| (*v - points[(i + 1) % n]).length())
        .fold(0.0_f32, f32::max)
        .max(1e-4);

    SeedShape {
        vertices: points,
        gap_ratio,
        mean_target: perimeter / n as f32,
        max_edge,
    }
}

#[derive(Clone, Copy, Debug)]
struct Placement {
    scale: f32,
    reflected: bool,
    angle: f32,
    joint: Vec2,
}

fn instance_vertices(seed: &[Vec2], placement: Placement) -> Vec<Vec2> {
    let a = seed[0] * placement.scale;
    let b = seed[1] * placement.scale;
    let mid = (a + b) * 0.5;
    seed.iter()
        .map(|vertex| {
            let mut point = *vertex * placement.scale;
            if placement.reflected {
                point = reflect_across(point, a, b);
            }
            rotate_around(point, mid, placement.angle) + (placement.joint - mid)
        })
        .collect()
}

fn root_vertices(seed: &[Vec2], rotation_rad: f32) -> Vec<Vec2> {
    seed.iter()
        .map(|vertex| rotate_around(*vertex, Vec2::ZERO, rotation_rad))
        .collect()
}

fn body_side(verts: &[Vec2]) -> f32 {
    let mid = (verts[0] + verts[1]) * 0.5;
    let base = verts[1] - verts[0];
    cross(base, centroid(verts) - mid)
}

fn placement_for_edge(
    seed: &[Vec2],
    scale: f32,
    edge_start: Vec2,
    edge_end: Vec2,
    parent_centroid: Vec2,
    inward: bool,
    twist_rad: f32,
    depth: u32,
) -> Placement {
    let scaled: Vec<Vec2> = seed.iter().map(|vertex| *vertex * scale).collect();
    let local_side = body_side(&scaled);
    let edge = edge_end - edge_start;
    let parent_side = cross(edge, parent_centroid - (edge_start + edge_end) * 0.5);
    let desired = if inward { parent_side } else { -parent_side };
    let reflected = local_side * desired < 0.0;
    let base = seed[1] - seed[0];
    let align = edge.y.atan2(edge.x) - base.y.atan2(base.x);
    Placement {
        scale,
        reflected,
        angle: align + twist_rad * depth as f32,
        joint: (edge_start + edge_end) * 0.5,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub depth: u16,
    pub color: [u8; 3],
    /// Tessellated pieces of one edge share a batch so they share a colour.
    pub batch: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Bounds {
    fn from_segments(segments: &[Segment]) -> Self {
        let mut bounds = Self {
            min_x: f32::MAX,
            min_y: f32::MAX,
            max_x: f32::MIN,
            max_y: f32::MIN,
        };
        if segments.is_empty() {
            return Self {
                min_x: -1.0,
                min_y: -1.0,
                max_x: 1.0,
                max_y: 1.0,
            };
        }
        for segment in segments {
            bounds.include(segment.x0, segment.y0);
            bounds.include(segment.x1, segment.y1);
        }
        bounds
    }

    fn include(&mut self, x: f32, y: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }
}

/// Ordered vertex ring of one polygon instance, including fillet samples.
#[derive(Clone, Debug)]
pub struct InstanceOutline {
    pub verts: Vec<Vec2>,
    pub depth: u16,
}

pub struct Output {
    pub segments: Vec<Segment>,
    /// Empty when no fill was requested, so the stroke path does no extra work.
    pub outlines: Vec<InstanceOutline>,
    pub bounds: Bounds,
    pub stroke_world: f32,
    pub truncated: bool,
}

struct Job {
    placement: Placement,
    depth: u32,
}

pub fn generate(params: &FractalParams, width: u32, height: u32) -> Output {
    let params = params.sanitized();
    let seed = close_seed(&params.side_lengths);
    let reach = conservative_reach(&seed, params.shrink, &params.shrink_by_depth, params.depth);
    let pixels_per_world = cull_pixels_per_world(width.max(1), height.max(1), reach);
    let min_world = 0.5 / pixels_per_world.max(1e-6);

    let mut root = root_vertices(&seed.vertices, params.rotation_deg.to_radians());
    if params.jitter > 0.0 {
        jitter_vertices(
            &mut root,
            params.jitter,
            params.jitter_seed,
            seed.max_edge,
            1.0,
            0,
            Vec2::ZERO,
        );
    }
    let emit_fills = params.fill_mode != FillMode::None;
    let mut segments = Vec::new();
    let mut outlines = Vec::new();
    let batches = AtomicU32::new(0);
    if let Some(verts) = push_outline(
        &root,
        0,
        params.corner_radius,
        pixels_per_world,
        &batches,
        &mut segments,
        emit_fills,
    ) {
        outlines.push(InstanceOutline { verts, depth: 0 });
    }

    let budget = AtomicUsize::new(segments.len());
    let truncated = AtomicBool::new(false);
    let twist = params.twist_deg.to_radians();

    let first_shrink = shrink_at(params.shrink, &params.shrink_by_depth, 1);
    if params.depth >= 1 && seed.max_edge * first_shrink >= min_world {
        let parent_centroid = centroid(&root);
        let mut jobs = Vec::new();
        let n = root.len();
        for i in 0..n {
            let start = root[i];
            let end = root[(i + 1) % n];
            if (end - start).length() < 1e-4 {
                continue;
            }
            jobs.push(Job {
                placement: placement_for_edge(
                    &seed.vertices,
                    first_shrink,
                    start,
                    end,
                    parent_centroid,
                    params.inward,
                    twist,
                    1,
                ),
                depth: 1,
            });
        }

        let parts: Vec<(Vec<Segment>, Vec<InstanceOutline>)> = jobs
            .into_par_iter()
            .map(|job| {
                grow_tree(
                    &seed,
                    &params,
                    job,
                    min_world,
                    twist,
                    pixels_per_world,
                    &budget,
                    &batches,
                    &truncated,
                    emit_fills,
                )
            })
            .collect();
        for (part, part_outlines) in parts {
            segments.extend(part);
            outlines.extend(part_outlines);
        }
    }

    paint_segments(&mut segments, &params);
    let stroke_world = params.stroke_fraction * seed.mean_target;
    let mut hit_limit = truncated.load(Ordering::Relaxed);
    replicate_segments(
        &mut segments,
        &mut outlines,
        params.symmetry,
        params.mirror,
        &mut hit_limit,
    );
    let bounds = Bounds::from_segments(&segments);
    Output {
        segments,
        outlines,
        bounds,
        stroke_world,
        truncated: hit_limit,
    }
}

/// One placed copy of a motif. `scale` is 1 for the root stamp.
#[derive(Clone, Copy, Debug)]
pub struct StampPlacement {
    pub center: Vec2,
    pub angle: f32,
    pub reflected: bool,
    pub scale: f32,
    pub depth: u16,
    pub motif_id: u64,
}

pub struct ArrangementOutput {
    pub placements: Vec<StampPlacement>,
    pub truncated: bool,
}

/// Walk the same edge-growth rule as [`generate`], recording a stamp pose per polygon.
pub fn arrange(params: &ArrangementParams, width: u32, height: u32) -> ArrangementOutput {
    let params = params.sanitized();
    let seed = close_seed(&params.side_lengths);
    let reach = conservative_reach(&seed, params.shrink, &params.shrink_by_depth, params.depth);
    let pixels_per_world = cull_pixels_per_world(width.max(1), height.max(1), reach);
    let min_world = 0.5 / pixels_per_world.max(1e-6);
    let rotation = params.rotation_deg.to_radians();
    let root = root_vertices(&seed.vertices, rotation);
    let mut placements = Vec::new();
    let mut truncated = false;
    placements.push(StampPlacement {
        center: centroid(&root),
        angle: rotation,
        reflected: false,
        scale: 1.0,
        depth: 0,
        motif_id: motif_id_at(&params, 0),
    });

    let twist = params.twist_deg.to_radians();
    let first_shrink = shrink_at(params.shrink, &params.shrink_by_depth, 1);
    if params.depth >= 1 && seed.max_edge * first_shrink >= min_world {
        let parent_centroid = centroid(&root);
        let mut stack = Vec::new();
        let n = root.len();
        for i in 0..n {
            let start = root[i];
            let end = root[(i + 1) % n];
            if (end - start).length() < 1e-4 {
                continue;
            }
            stack.push((
                Job {
                    placement: placement_for_edge(
                        &seed.vertices,
                        first_shrink,
                        start,
                        end,
                        parent_centroid,
                        params.inward,
                        twist,
                        1,
                    ),
                    depth: 1,
                },
                parent_centroid,
            ));
        }
        while let Some((job, parent_center)) = stack.pop() {
            if placements.len() >= MAX_INSTANCES {
                truncated = true;
                break;
            }
            let mut verts = instance_vertices(&seed.vertices, job.placement);
            let shift = layer_shift(centroid(&verts), parent_center, params.gap);
            for vertex in &mut verts {
                *vertex += shift;
            }
            let n = verts.len();
            placements.push(StampPlacement {
                center: centroid(&verts),
                angle: job.placement.angle,
                reflected: job.placement.reflected,
                scale: job.placement.scale,
                depth: job.depth as u16,
                motif_id: motif_id_at(&params, job.depth),
            });
            if job.depth >= params.depth {
                continue;
            }
            let next_scale = job.placement.scale
                * shrink_at(params.shrink, &params.shrink_by_depth, job.depth + 1);
            if seed.max_edge * next_scale < min_world {
                continue;
            }
            let parent_centroid = centroid(&verts);
            for i in 1..n {
                if placements.len() + stack.len() >= MAX_INSTANCES {
                    truncated = true;
                    break;
                }
                let start = verts[i];
                let end = verts[(i + 1) % n];
                if (end - start).length() < 1e-4 {
                    continue;
                }
                stack.push((
                    Job {
                        placement: placement_for_edge(
                            &seed.vertices,
                            next_scale,
                            start,
                            end,
                            parent_centroid,
                            params.inward,
                            twist,
                            job.depth + 1,
                        ),
                        depth: job.depth + 1,
                    },
                    parent_centroid,
                ));
            }
        }
    }

    replicate_placements(
        &mut placements,
        params.symmetry,
        params.mirror,
        &mut truncated,
    );

    ArrangementOutput {
        placements,
        truncated,
    }
}

fn replicate_segments(
    segments: &mut Vec<Segment>,
    outlines: &mut Vec<InstanceOutline>,
    symmetry: u32,
    mirror: bool,
    truncated: &mut bool,
) {
    if symmetry <= 1 && !mirror {
        return;
    }
    let symmetry = symmetry.max(1);
    let base_len = segments.len();
    if base_len == 0 {
        return;
    }
    let base_outlines = outlines.len();
    for k in 0..symmetry {
        let angle = std::f32::consts::TAU * k as f32 / symmetry as f32;
        if k > 0 && !push_segment_copy(segments, base_len, angle, false, truncated) {
            return;
        }
        if k > 0 {
            push_outline_copy(outlines, base_outlines, angle, false);
        }
        if mirror && !push_segment_copy(segments, base_len, angle, true, truncated) {
            return;
        }
        if mirror {
            push_outline_copy(outlines, base_outlines, angle, true);
        }
    }
}

fn push_segment_copy(
    segments: &mut Vec<Segment>,
    base_len: usize,
    angle: f32,
    mirror: bool,
    truncated: &mut bool,
) -> bool {
    if segments.len() + base_len > MAX_SEGMENTS {
        *truncated = true;
        return false;
    }
    for index in 0..base_len {
        let segment = segments[index];
        segments.push(transform_segment(segment, angle, mirror));
    }
    true
}

fn push_outline_copy(
    outlines: &mut Vec<InstanceOutline>,
    base_len: usize,
    angle: f32,
    mirror: bool,
) {
    if base_len == 0 {
        return;
    }
    for index in 0..base_len {
        let copy = transform_outline(&outlines[index], angle, mirror);
        outlines.push(copy);
    }
}

fn transform_outline(outline: &InstanceOutline, angle: f32, mirror: bool) -> InstanceOutline {
    InstanceOutline {
        verts: outline
            .verts
            .iter()
            .copied()
            .map(|point| transform_point(point, angle, mirror))
            .collect(),
        depth: outline.depth,
    }
}

fn transform_point(mut point: Vec2, angle: f32, mirror: bool) -> Vec2 {
    if mirror {
        point = reflect_across(point, Vec2::ZERO, Vec2::new(1.0, 0.0));
    }
    if angle != 0.0 {
        point = rotate_around(point, Vec2::ZERO, angle);
    }
    point
}

fn transform_segment(segment: Segment, angle: f32, mirror: bool) -> Segment {
    let mut start = Vec2::new(segment.x0, segment.y0);
    let mut end = Vec2::new(segment.x1, segment.y1);
    if mirror {
        start = reflect_across(start, Vec2::ZERO, Vec2::new(1.0, 0.0));
        end = reflect_across(end, Vec2::ZERO, Vec2::new(1.0, 0.0));
    }
    if angle != 0.0 {
        start = rotate_around(start, Vec2::ZERO, angle);
        end = rotate_around(end, Vec2::ZERO, angle);
    }
    Segment {
        x0: start.x,
        y0: start.y,
        x1: end.x,
        y1: end.y,
        ..segment
    }
}

fn replicate_placements(
    placements: &mut Vec<StampPlacement>,
    symmetry: u32,
    mirror: bool,
    truncated: &mut bool,
) {
    if symmetry <= 1 && !mirror {
        return;
    }
    let symmetry = symmetry.max(1);
    let base_len = placements.len();
    if base_len == 0 {
        return;
    }
    for k in 0..symmetry {
        let angle = std::f32::consts::TAU * k as f32 / symmetry as f32;
        if k > 0 && !push_placement_copy(placements, base_len, angle, false, truncated) {
            return;
        }
        if mirror && !push_placement_copy(placements, base_len, angle, true, truncated) {
            return;
        }
    }
}

fn push_placement_copy(
    placements: &mut Vec<StampPlacement>,
    base_len: usize,
    angle: f32,
    mirror: bool,
    truncated: &mut bool,
) -> bool {
    if placements.len() + base_len > MAX_INSTANCES {
        *truncated = true;
        return false;
    }
    for index in 0..base_len {
        let placement = placements[index];
        placements.push(transform_placement(placement, angle, mirror));
    }
    true
}

fn transform_placement(placement: StampPlacement, angle: f32, mirror: bool) -> StampPlacement {
    let mut center = placement.center;
    let mut theta = placement.angle;
    let mut reflected = placement.reflected;
    if mirror {
        center = reflect_across(center, Vec2::ZERO, Vec2::new(1.0, 0.0));
        theta = -theta;
        reflected = !reflected;
    }
    if angle != 0.0 {
        center = rotate_around(center, Vec2::ZERO, angle);
        theta += angle;
    }
    StampPlacement {
        center,
        angle: theta,
        reflected,
        ..placement
    }
}

fn layer_shift(child: Vec2, parent: Vec2, gap: f32) -> Vec2 {
    if gap.abs() < 1e-6 {
        return Vec2::ZERO;
    }
    let away = child - parent;
    // Positive values add whole parent-to-child steps.
    // Negative values close that gap and stop on the parent, so -4 does not
    // fling the layer out the other side.
    let t = if gap >= 0.0 {
        gap
    } else {
        (gap / GAP_LIMIT).clamp(-1.0, 0.0)
    };
    away * t
}

fn motif_id_at(params: &ArrangementParams, depth: u32) -> u64 {
    params
        .motif_by_depth
        .get(depth as usize)
        .copied()
        .unwrap_or_else(|| params.motif_by_depth.first().copied().unwrap_or(0))
}

/// Move each vertex by `jitter * max_edge * r` in seed space, with `r` in [-1, 1].
/// The instance scale carries that offset into world space, so each copy jitters by a
/// fraction of its own edge. Callers skip this when `jitter` is 0 so positions stay exact.
fn jitter_vertices(
    verts: &mut [Vec2],
    jitter: f32,
    jitter_seed: u64,
    max_edge: f32,
    scale: f32,
    depth: u32,
    joint: Vec2,
) {
    let amplitude = jitter * max_edge * scale;
    if !(amplitude > 0.0) {
        return;
    }
    for (index, vertex) in verts.iter_mut().enumerate() {
        *vertex += vertex_jitter(jitter_seed, depth, joint, index, amplitude);
    }
}

fn vertex_jitter(jitter_seed: u64, depth: u32, joint: Vec2, index: usize, amplitude: f32) -> Vec2 {
    let mut rng = ColourRng(jitter_state(jitter_seed, depth, joint, index));
    let r = rng.gen_range_f32(-1.0, 1.0);
    let (sin, cos) = rng.gen_range_f32(0.0, std::f32::consts::TAU).sin_cos();
    Vec2::new(cos, sin) * (amplitude * r)
}

fn jitter_state(jitter_seed: u64, depth: u32, joint: Vec2, index: usize) -> u64 {
    let mut state = jitter_seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (depth as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = state
        .wrapping_mul(0x94D0_49BB_1331_11EB)
        .wrapping_add(joint.x.to_bits() as u64);
    state = state
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(joint.y.to_bits() as u64);
    state
        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
        .wrapping_add(index as u64)
}

fn grow_tree(
    seed: &SeedShape,
    params: &FractalParams,
    root_job: Job,
    min_world: f32,
    twist: f32,
    pixels_per_world: f32,
    budget: &AtomicUsize,
    batches: &AtomicU32,
    truncated: &AtomicBool,
    emit_fills: bool,
) -> (Vec<Segment>, Vec<InstanceOutline>) {
    let mut segments = Vec::new();
    let mut outlines = Vec::new();
    let mut stack = vec![root_job];
    while let Some(job) = stack.pop() {
        let mut verts = instance_vertices(&seed.vertices, job.placement);
        if params.jitter > 0.0 {
            jitter_vertices(
                &mut verts,
                params.jitter,
                params.jitter_seed,
                seed.max_edge,
                job.placement.scale,
                job.depth,
                job.placement.joint,
            );
        }
        let n = verts.len();
        let mut outline = Vec::new();
        let ring = push_outline(
            &verts,
            job.depth,
            params.corner_radius,
            pixels_per_world,
            batches,
            &mut outline,
            emit_fills,
        );
        let added = outline.len();
        let used = budget.fetch_add(added, Ordering::Relaxed);
        if used + added > MAX_SEGMENTS {
            truncated.store(true, Ordering::Relaxed);
            break;
        }
        segments.extend(outline);
        if let Some(verts) = ring {
            outlines.push(InstanceOutline {
                verts,
                depth: job.depth as u16,
            });
        }
        if job.depth >= params.depth {
            continue;
        }
        let next_scale = job.placement.scale
            * shrink_at(params.shrink, &params.shrink_by_depth, job.depth + 1);
        if seed.max_edge * next_scale < min_world {
            continue;
        }
        let parent_centroid = centroid(&verts);
        for i in 1..n {
            let start = verts[i];
            let end = verts[(i + 1) % n];
            if (end - start).length() < 1e-4 {
                continue;
            }
            stack.push(Job {
                placement: placement_for_edge(
                    &seed.vertices,
                    next_scale,
                    start,
                    end,
                    parent_centroid,
                    params.inward,
                    twist,
                    job.depth + 1,
                ),
                depth: job.depth + 1,
            });
        }
    }
    (segments, outlines)
}

fn push_outline(
    verts: &[Vec2],
    depth: u32,
    corner_radius: f32,
    pixels_per_world: f32,
    batches: &AtomicU32,
    out: &mut Vec<Segment>,
    emit_ring: bool,
) -> Option<Vec<Vec2>> {
    let n = verts.len();
    if n < 2 {
        return None;
    }
    // Rings are built only for fills. The segment loop below stays the stroke path.
    let mut ring = if emit_ring { Some(Vec::new()) } else { None };
    if corner_radius <= 1e-4 || n < 3 {
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            push_piece(out, a, b, depth, 0);
        }
        if let Some(ring) = ring.as_mut() {
            ring.extend_from_slice(verts);
        }
        return finish_ring(ring);
    }
    let cuts = corner_cuts(verts, corner_radius, pixels_per_world);
    for i in 0..n {
        let next = (i + 1) % n;
        let from = cuts[i].leave;
        let to = cuts[next].arrive;
        let batch = batches.fetch_add(1, Ordering::Relaxed);
        push_piece(out, from, to, depth, batch);
        if let Some(ring) = ring.as_mut() {
            push_ring_point(ring, from);
            push_ring_point(ring, to);
        }
        let mut prev = to;
        for point in &cuts[next].arc {
            push_piece(out, prev, *point, depth, batch);
            if let Some(ring) = ring.as_mut() {
                push_ring_point(ring, *point);
            }
            prev = *point;
        }
    }
    finish_ring(ring)
}

fn push_ring_point(ring: &mut Vec<Vec2>, point: Vec2) {
    if let Some(prev) = ring.last().copied() {
        if (prev - point).length() < 1e-6 {
            return;
        }
    }
    ring.push(point);
}

fn finish_ring(ring: Option<Vec<Vec2>>) -> Option<Vec<Vec2>> {
    let mut ring = ring?;
    if ring.len() >= 2 {
        let first = ring[0];
        let last = ring[ring.len() - 1];
        if (first - last).length() < 1e-5 {
            ring.pop();
        }
    }
    if ring.len() >= 3 {
        Some(ring)
    } else {
        None
    }
}

struct CornerCut {
    /// Where the outline leaves this vertex along the next edge.
    leave: Vec2,
    /// Where the previous edge arrives at this vertex.
    arrive: Vec2,
    /// Samples after `arrive`, ending at `leave`.
    arc: Vec<Vec2>,
}

fn corner_cuts(verts: &[Vec2], radius_fraction: f32, pixels_per_world: f32) -> Vec<CornerCut> {
    let n = verts.len();
    let mut cuts = Vec::with_capacity(n);
    for i in 0..n {
        let prev = verts[(i + n - 1) % n];
        let vertex = verts[i];
        let next = verts[(i + 1) % n];
        cuts.push(fillet_corner(
            prev,
            vertex,
            next,
            radius_fraction,
            pixels_per_world,
        ));
    }
    cuts
}

fn sharp_corner(vertex: Vec2) -> CornerCut {
    CornerCut {
        leave: vertex,
        arrive: vertex,
        arc: Vec::new(),
    }
}

fn fillet_corner(
    prev: Vec2,
    vertex: Vec2,
    next: Vec2,
    radius_fraction: f32,
    pixels_per_world: f32,
) -> CornerCut {
    let incoming = vertex - prev;
    let outgoing = next - vertex;
    let len_in = incoming.length();
    let len_out = outgoing.length();
    if len_in < 1e-6 || len_out < 1e-6 || radius_fraction <= 1e-4 {
        return sharp_corner(vertex);
    }
    // Reflected layers are clockwise. Both windings get a fillet.
    if cross(incoming, outgoing).abs() <= 1e-5 {
        return sharp_corner(vertex);
    }
    let Some(toward_prev) = normalized(prev - vertex) else {
        return sharp_corner(vertex);
    };
    let Some(toward_next) = normalized(next - vertex) else {
        return sharp_corner(vertex);
    };
    let angle = toward_prev.dot(toward_next).clamp(-1.0, 1.0).acos();
    if !(0.15..=3.0).contains(&angle) {
        return sharp_corner(vertex);
    }
    let half = angle * 0.5;
    let tan_half = half.tan();
    let sin_half = half.sin();
    if tan_half < 1e-4 || sin_half < 1e-4 {
        return sharp_corner(vertex);
    }
    let tangent = radius_fraction * 0.45 * len_in.min(len_out);
    if tangent < 1e-5 {
        return sharp_corner(vertex);
    }
    let radius = tangent * tan_half;
    let Some(bisector) = normalized(toward_prev + toward_next) else {
        return sharp_corner(vertex);
    };
    let center = vertex + bisector * (radius / sin_half);
    let arrive = vertex + toward_prev * tangent;
    let leave = vertex + toward_next * tangent;
    let start = (arrive.y - center.y).atan2(arrive.x - center.x);
    let end = (leave.y - center.y).atan2(leave.x - center.x);
    let mut sweep = end - start;
    const PI: f32 = std::f32::consts::PI;
    while sweep <= -PI {
        sweep += std::f32::consts::TAU;
    }
    while sweep > PI {
        sweep -= std::f32::consts::TAU;
    }
    let expected = PI - angle;
    if sweep.abs() < 0.05 || (sweep.abs() - expected).abs() > 0.35 {
        return sharp_corner(vertex);
    }
    let Some(steps) = arc_step_count(radius, sweep, pixels_per_world) else {
        return sharp_corner(vertex);
    };
    let mut arc = Vec::with_capacity(steps);
    for step in 1..=steps {
        let t = step as f32 / steps as f32;
        let ang = start + sweep * t;
        arc.push(center + Vec2::new(ang.cos(), ang.sin()) * radius);
    }
    if let Some(last) = arc.last_mut() {
        *last = leave;
    }
    CornerCut {
        leave,
        arrive,
        arc,
    }
}

fn arc_step_count(radius: f32, sweep: f32, pixels_per_world: f32) -> Option<usize> {
    let pixels = radius * sweep.abs() * pixels_per_world.max(0.0);
    if pixels < 1.0 {
        None
    } else {
        Some(((pixels / 4.0).round() as usize).clamp(1, 8))
    }
}

fn normalized(vector: Vec2) -> Option<Vec2> {
    let len = vector.length();
    if len < 1e-8 {
        None
    } else {
        Some(vector * (1.0 / len))
    }
}

fn push_piece(out: &mut Vec<Segment>, a: Vec2, b: Vec2, depth: u32, batch: u32) {
    if (b - a).length() < 1e-6 {
        return;
    }
    out.push(Segment {
        x0: a.x,
        y0: a.y,
        x1: b.x,
        y1: b.y,
        depth: depth as u16,
        color: [0, 0, 0],
        batch,
    });
}

/// Shrink applied when creating a polygon at `depth`. A missing entry uses `base`.
fn shrink_at(base: f32, shrink_by_depth: &[f32], depth: u32) -> f32 {
    match shrink_by_depth.get(depth as usize).copied() {
        Some(value) if value.is_finite() => value.clamp(0.05, 0.95),
        _ => base,
    }
}

fn conservative_reach(seed: &SeedShape, shrink: f32, shrink_by_depth: &[f32], depth: u32) -> f32 {
    let radius = seed
        .vertices
        .iter()
        .map(|vertex| vertex.length())
        .fold(0.0_f32, f32::max)
        .max(1e-3);
    if depth == 0 {
        return radius;
    }
    let mid = (seed.vertices[0] + seed.vertices[1]) * 0.5;
    let joint_radius = seed
        .vertices
        .iter()
        .map(|vertex| (*vertex - mid).length())
        .fold(0.0_f32, f32::max)
        .max(1e-3);
    // An empty schedule, or one that repeats `shrink`, must keep the old closed form
    // so culling thresholds stay bit-identical.
    let uniform = (1..=depth).all(|level| shrink_at(shrink, shrink_by_depth, level) == shrink);
    let extra = if uniform {
        if (shrink - 1.0).abs() < 1e-4 {
            joint_radius * depth as f32
        } else {
            let s = shrink;
            joint_radius * s * (1.0 - s.powi(depth as i32)) / (1.0 - s)
        }
    } else {
        let mut scale = 1.0;
        let mut sum = 0.0;
        for level in 1..=depth {
            scale *= shrink_at(shrink, shrink_by_depth, level);
            sum += scale;
        }
        joint_radius * sum
    };
    (radius + extra).max(1e-3)
}

fn cull_pixels_per_world(width: u32, height: u32, reach: f32) -> f32 {
    let avail = (width.min(height) as f32) * (1.0 - 2.0 * MARGIN);
    avail / (2.0 * reach)
}

fn paint_segments(segments: &mut [Segment], params: &FractalParams) {
    let mut rng = ColourRng(params.color_seed | 1);
    if params.corner_radius <= 1e-4 {
        for segment in segments.iter_mut() {
            segment.color = colour_for(params, segment.depth as u32, &mut rng);
        }
        return;
    }
    let mut start = 0;
    while start < segments.len() {
        let batch = segments[start].batch;
        let mut end = start + 1;
        while end < segments.len() && segments[end].batch == batch {
            end += 1;
        }
        let color = colour_for(params, segments[start].depth as u32, &mut rng);
        for segment in &mut segments[start..end] {
            segment.color = color;
        }
        start = end;
    }
}

fn colour_for(params: &FractalParams, depth: u32, rng: &mut ColourRng) -> [u8; 3] {
    match params.colour_mode {
        ColourMode::Single => params.color_start,
        ColourMode::Gradient => depth_color(params, depth),
        ColourMode::Random => rng.next_colour(),
        ColourMode::Palette => palette_color(params, depth),
    }
}

struct ColourRng(u64);

impl ColourRng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }

    fn next_colour(&mut self) -> [u8; 3] {
        let value = self.next_u32();
        [
            (value & 255) as u8,
            ((value >> 8) & 255) as u8,
            ((value >> 16) & 255) as u8,
        ]
    }

    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    fn gen_bool(&mut self) -> bool {
        self.next_u32() & 1 == 1
    }

    fn gen_range_u32(&mut self, min: u32, max: u32) -> u32 {
        min + self.next_u32() % (max - min + 1)
    }

    fn gen_range_f32(&mut self, min: f32, max: f32) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        min + (max - min) * unit
    }
}

/// RGB channels in 0..=255 to hue (degrees), saturation, and value, each hue in 0..360 and S/V in 0..=1.
fn rgb_to_hsv(rgb: [u8; 3]) -> [f32; 3] {
    let r = rgb[0] as f32 / 255.0;
    let g = rgb[1] as f32 / 255.0;
    let b = rgb[2] as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let saturation = if max == 0.0 { 0.0 } else { delta / max };
    [hue, saturation, max]
}

/// Hue in degrees, saturation and value in 0..=1, back to 8-bit RGB.
fn hsv_to_rgb(hsv: [f32; 3]) -> [u8; 3] {
    let hue = hsv[0].rem_euclid(360.0);
    let saturation = hsv[1].clamp(0.0, 1.0);
    let value = hsv[2].clamp(0.0, 1.0);
    let chroma = value * saturation;
    let hue_sector = hue / 60.0;
    let x = chroma * (1.0 - ((hue_sector % 2.0) - 1.0).abs());
    let m = value - chroma;
    let (red, green, blue) = match hue_sector as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let channel = |component: f32| ((component + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    [channel(red), channel(green), channel(blue)]
}

/// Blend `start` into `end` by `t`. `hsv` uses the shortest hue path; otherwise each RGB channel.
fn lerp_colour(start: [u8; 3], end: [u8; 3], t: f32, hsv: bool) -> [u8; 3] {
    if hsv {
        let start = rgb_to_hsv(start);
        let end = rgb_to_hsv(end);
        let mut hue_delta = end[0] - start[0];
        if hue_delta > 180.0 {
            hue_delta -= 360.0;
        } else if hue_delta < -180.0 {
            hue_delta += 360.0;
        }
        hsv_to_rgb([
            (start[0] + hue_delta * t).rem_euclid(360.0),
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        ])
    } else {
        let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        [mix(start[0], end[0]), mix(start[1], end[1]), mix(start[2], end[2])]
    }
}

pub fn depth_color(params: &FractalParams, depth: u32) -> [u8; 3] {
    let max_depth = params.depth.max(1) as f32;
    let t = (depth as f32 / max_depth).clamp(0.0, 1.0);
    lerp_colour(params.color_start, params.color_end, t, params.gradient_hsv)
}

/// Fill colour at `depth`. Solid uses `color_start`. Palette reuses the depth sample.
pub fn fill_color(params: &FractalParams, depth: u32) -> [u8; 3] {
    match params.fill_mode {
        FillMode::None | FillMode::Solid => params.color_start,
        FillMode::Palette => palette_color(params, depth),
    }
}

/// Sample `palette` at `t = depth/max_depth`, linearly between adjacent stops.
fn palette_color(params: &FractalParams, depth: u32) -> [u8; 3] {
    let stops = params.palette.as_slice();
    match stops {
        [] => depth_color(params, depth),
        [only] => *only,
        stops => {
            let max_depth = params.depth.max(1) as f32;
            let t = (depth as f32 / max_depth).clamp(0.0, 1.0);
            let scaled = t * (stops.len() - 1) as f32;
            let index = (scaled.floor() as usize).min(stops.len() - 2);
            let frac = scaled - index as f32;
            lerp_colour(stops[index], stops[index + 1], frac, params.gradient_hsv)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_sum(verts: &[Vec2]) -> Vec2 {
        let mut sum = Vec2::ZERO;
        let n = verts.len();
        for i in 0..n {
            sum += verts[(i + 1) % n] - verts[i];
        }
        sum
    }

    #[test]
    fn equal_lengths_match_a_regular_polygon() {
        let seed = close_seed(&[1.0, 1.0, 1.0, 1.0]);
        assert!(seed.gap_ratio < 1e-4, "gap {}", seed.gap_ratio);
        let n = seed.vertices.len();
        let radius = 1.0 / (2.0 * (std::f32::consts::PI / n as f32).sin());
        for vertex in &seed.vertices {
            let distance = vertex.length();
            assert!((distance - radius).abs() / radius < 1e-3, "{distance} vs {radius}");
        }
        for i in 0..n {
            let length = (seed.vertices[(i + 1) % n] - seed.vertices[i]).length();
            assert!((length - 1.0).abs() < 1e-3, "{length}");
        }
        assert!(edge_sum(&seed.vertices).length() < 1e-4);
    }

    #[test]
    fn unequal_lengths_meet_after_the_gap_is_spread() {
        let lengths = [1.0, 2.0, 0.5, 1.5, 1.2];
        let seed = close_seed(&lengths);
        assert!(seed.gap_ratio > 1e-3, "expected a real gap, got {}", seed.gap_ratio);
        assert!(
            edge_sum(&seed.vertices).length() < 1e-4,
            "outline did not close"
        );
    }

    #[test]
    fn child_joint_lies_on_the_parent_edge_when_twist_is_zero() {
        let seed = close_seed(&[1.0, 1.2, 0.8, 1.1]);
        let root = root_vertices(&seed.vertices, 20.0_f32.to_radians());
        let parent_centroid = centroid(&root);
        let n = root.len();
        for i in 0..n {
            let start = root[i];
            let end = root[(i + 1) % n];
            let placement = placement_for_edge(
                &seed.vertices,
                0.5,
                start,
                end,
                parent_centroid,
                false,
                0.0,
                1,
            );
            let child = instance_vertices(&seed.vertices, placement);
            let child_mid = (child[0] + child[1]) * 0.5;
            let parent_mid = (start + end) * 0.5;
            assert!(
                (child_mid - parent_mid).length() < 1e-4,
                "midpoint drifted by {}",
                (child_mid - parent_mid).length()
            );
            let base = child[1] - child[0];
            let edge = end - start;
            let sine = cross(base, edge).abs() / (base.length() * edge.length());
            assert!(sine < 1e-4, "sides are not parallel, sine {sine}");
            assert!(base.dot(edge) > 0.0, "sides face opposite directions");
        }
    }

    #[test]
    fn inward_centroid_is_opposite_outward() {
        let seed = close_seed(&[1.0, 1.0, 1.0]);
        let root = root_vertices(&seed.vertices, 0.0);
        let parent_centroid = centroid(&root);
        let start = root[0];
        let end = root[1];
        let outward = instance_vertices(
            &seed.vertices,
            placement_for_edge(
                &seed.vertices,
                0.5,
                start,
                end,
                parent_centroid,
                false,
                0.0,
                1,
            ),
        );
        let inward = instance_vertices(
            &seed.vertices,
            placement_for_edge(
                &seed.vertices,
                0.5,
                start,
                end,
                parent_centroid,
                true,
                0.0,
                1,
            ),
        );
        let edge = end - start;
        let mid = (start + end) * 0.5;
        let parent_side = cross(edge, parent_centroid - mid);
        let outward_side = cross(edge, centroid(&outward) - mid);
        let inward_side = cross(edge, centroid(&inward) - mid);
        assert!(
            outward_side * inward_side < 0.0,
            "out {outward_side} in {inward_side}"
        );
        assert!(
            outward_side * parent_side < 0.0,
            "outward child should sit on the far side of the edge"
        );
        assert!(
            inward_side * parent_side > 0.0,
            "inward child should sit on the parent's side of the edge"
        );
    }

    #[test]
    fn non_root_does_not_grow_from_the_joint() {
        let mut params = FractalParams::default();
        params.depth = 2;
        params.shrink = 0.5;
        let output = generate(&params, 4096, 4096);
        let count = |depth| output.segments.iter().filter(|s| s.depth == depth).count();
        assert_eq!(count(0), 3);
        assert_eq!(count(1), 9);
        assert_eq!(count(2), 18);
    }

    #[test]
    fn colour_modes_paint_each_line() {
        let mut params = FractalParams::default();
        params.depth = 2;
        params.colour_mode = ColourMode::Single;
        params.color_start = [10, 20, 30];
        let single = generate(&params, 512, 512);
        assert!(single.segments.iter().all(|segment| segment.color == [10, 20, 30]));

        params.colour_mode = ColourMode::Gradient;
        params.color_start = [0, 0, 0];
        params.color_end = [255, 255, 255];
        let gradient = generate(&params, 512, 512);
        let start = gradient
            .segments
            .iter()
            .find(|segment| segment.depth == 0)
            .unwrap()
            .color;
        let end = gradient
            .segments
            .iter()
            .find(|segment| segment.depth == 2)
            .unwrap()
            .color;
        assert_eq!(start, [0, 0, 0]);
        assert_eq!(end, [255, 255, 255]);

        params.colour_mode = ColourMode::Random;
        params.color_seed = 42;
        let random = generate(&params, 512, 512);
        let mut colours = std::collections::HashSet::new();
        for segment in &random.segments {
            colours.insert(segment.color);
        }
        assert!(colours.len() > 1, "expected a different colour on different lines");
    }

    #[test]
    fn hsv_gradient_midpoint_differs_from_rgb_lerp() {
        let mut params = FractalParams::default();
        assert!(!params.gradient_hsv);
        params.depth = 2;
        params.color_start = [255, 0, 0];
        params.color_end = [0, 255, 0];
        let rgb = depth_color(&params, 1);
        assert_eq!(rgb, [128, 128, 0]);
        assert_eq!(params.sanitized().gradient_hsv, false);

        params.gradient_hsv = true;
        let hsv = depth_color(&params.sanitized(), 1);
        assert_ne!(hsv, rgb);
        assert_eq!(hsv, [255, 255, 0]);
    }

    #[test]
    fn palette_mode_interpolates_across_depths() {
        let mut params = FractalParams::default();
        assert!(params.palette.len() >= 2);
        params.depth = 4;
        params.colour_mode = ColourMode::Palette;
        params.gradient_hsv = false;
        params.palette = vec![[0, 0, 0], [0, 0, 255], [255, 0, 0]];

        let output = generate(&params, 1024, 1024);
        let color_at = |depth: u16| {
            output
                .segments
                .iter()
                .find(|segment| segment.depth == depth)
                .unwrap()
                .color
        };
        assert_eq!(color_at(0), [0, 0, 0]);
        assert_eq!(color_at(1), [0, 0, 128]);
        assert_eq!(color_at(2), [0, 0, 255]);
        assert_eq!(color_at(3), [128, 0, 128]);
        assert_eq!(color_at(4), [255, 0, 0]);

        params.gradient_hsv = true;
        params.palette = vec![[255, 0, 0], [0, 255, 0]];
        let hsv = generate(&params, 1024, 1024);
        let mid = hsv
            .segments
            .iter()
            .find(|segment| segment.depth == 2)
            .unwrap()
            .color;
        assert_eq!(mid, [255, 255, 0]);

        params.palette.clear();
        params.color_start = [1, 2, 3];
        params.color_end = [4, 5, 6];
        assert_eq!(
            params.sanitized().palette,
            vec![[1, 2, 3], [4, 5, 6]]
        );
        params.palette = vec![[9, 9, 9]];
        assert_eq!(params.sanitized().palette.len(), 2);
    }

    #[test]
    fn tiny_output_culls_deep_levels() {
        let mut params = FractalParams::default();
        params.depth = 8;
        params.shrink = 0.5;
        let big = generate(&params, 1024, 1024);
        let tiny = generate(&params, 24, 24);
        let max_depth = |output: &Output| output.segments.iter().map(|s| s.depth).max().unwrap();
        assert!(tiny.segments.len() < big.segments.len());
        assert!(max_depth(&tiny) < max_depth(&big));
    }

    #[test]
    fn zero_radius_matches_the_sharp_segment_list() {
        let mut params = FractalParams::default();
        params.side_lengths = vec![1.0, 1.6, 0.7, 1.2];
        params.depth = 2;
        params.shrink = 0.5;
        params.rotation_deg = 20.0;
        params.twist_deg = -12.0;
        params.corner_radius = 0.0;
        let output = generate(&params, 1024, 1024);
        let seed = close_seed(&params.side_lengths);
        let root = root_vertices(&seed.vertices, params.rotation_deg.to_radians());
        let root_segments: Vec<_> = output.segments.iter().filter(|s| s.depth == 0).collect();
        assert_eq!(root_segments.len(), root.len());
        for (index, segment) in root_segments.iter().enumerate() {
            let start = root[index];
            let end = root[(index + 1) % root.len()];
            assert!((segment.x0 - start.x).abs() < 1e-5 && (segment.y0 - start.y).abs() < 1e-5);
            assert!((segment.x1 - end.x).abs() < 1e-5 && (segment.y1 - end.y).abs() < 1e-5);
        }
        // Quad: 1 root, 4 children, 12 grandchildren, four edges each.
        assert_eq!(output.segments.len(), 4 + 16 + 48);
    }

    #[test]
    fn positive_radius_fillets_inside_the_sharp_corner() {
        let prev = Vec2::new(0.0, 0.0);
        let vertex = Vec2::new(1.0, 0.0);
        let next = Vec2::new(1.0, 1.0);
        let cut = fillet_corner(prev, vertex, next, 1.0, 200.0);
        assert!(!cut.arc.is_empty());
        assert!((cut.arrive.x - 0.55).abs() < 1e-3, "arrive {}", cut.arrive.x);
        assert!(cut.arrive.y.abs() < 1e-3);
        assert!((cut.leave.x - 1.0).abs() < 1e-3);
        assert!((cut.leave.y - 0.45).abs() < 1e-3, "leave {}", cut.leave.y);
        for point in &cut.arc {
            assert!(point.x <= 1.0 + 1e-4 && point.y >= -1e-4);
            assert!(point.x < 1.0 - 1e-3 || point.y > 1e-3);
        }

        let mut sharp_params = FractalParams::default();
        sharp_params.depth = 2;
        sharp_params.corner_radius = 0.0;
        let mut rounded_params = sharp_params.clone();
        rounded_params.corner_radius = 1.0;
        let sharp = generate(&sharp_params, 1024, 1024);
        let rounded = generate(&rounded_params, 1024, 1024);
        let sharp_root: Vec<Vec2> = sharp
            .segments
            .iter()
            .filter(|segment| segment.depth == 0)
            .map(|segment| Vec2::new(segment.x0, segment.y0))
            .collect();
        let rounded_root: Vec<_> = rounded
            .segments
            .iter()
            .filter(|segment| segment.depth == 0)
            .collect();
        assert!(rounded_root.len() > sharp_root.len());
        assert!(rounded.segments.len() > sharp.segments.len());
        for window in rounded_root.windows(2) {
            assert!((window[0].x1 - window[1].x0).abs() < 1e-3);
            assert!((window[0].y1 - window[1].y0).abs() < 1e-3);
        }
        let first = rounded_root[0];
        let last = rounded_root[rounded_root.len() - 1];
        assert!((last.x1 - first.x0).abs() < 1e-3 && (last.y1 - first.y0).abs() < 1e-3);
        for segment in &rounded_root {
            assert!(point_inside_ccw(Vec2::new(segment.x0, segment.y0), &sharp_root));
            assert!(point_inside_ccw(Vec2::new(segment.x1, segment.y1), &sharp_root));
        }
    }

    fn point_inside_ccw(point: Vec2, verts: &[Vec2]) -> bool {
        let n = verts.len();
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            if cross(b - a, point - a) < -1e-3 {
                return false;
            }
        }
        true
    }

    #[test]
    fn triangle_arrangement_counts_follow_the_edge_rule() {
        let mut params = ArrangementParams::default();
        params.depth = 2;
        params.motif_by_depth = vec![7, 7, 7];
        let output = arrange(&params, 512, 512);
        let count = |depth| {
            output
                .placements
                .iter()
                .filter(|placement| placement.depth == depth)
                .count()
        };
        assert_eq!(count(0), 1);
        assert_eq!(count(1), 3);
        assert_eq!(count(2), 6);
        assert!(output
            .placements
            .iter()
            .all(|placement| placement.motif_id == 7));
        assert!(!output.truncated);

        params.depth = 0;
        params.motif_by_depth = vec![7];
        let root_only = arrange(&params, 512, 512);
        assert_eq!(root_only.placements.len(), 1);
        assert_eq!(root_only.placements[0].scale, 1.0);
    }

    #[test]
    fn gap_moves_layers_closer_or_further_apart() {
        let mut params = ArrangementParams::default();
        params.depth = 1;
        params.shrink = 0.5;
        params.gap = 0.0;
        let baseline = arrange(&params, 512, 512);
        params.gap = 1.0;
        let further = arrange(&params, 512, 512);
        params.gap = -2.0;
        let closer = arrange(&params, 512, 512);

        let root = baseline.placements[0].center;
        assert!((further.placements[0].center - root).length() < 1e-5);
        assert!((closer.placements[0].center - root).length() < 1e-5);

        let distance = |output: &ArrangementOutput| {
            let children: Vec<_> = output
                .placements
                .iter()
                .filter(|placement| placement.depth == 1)
                .collect();
            let sum: f32 = children
                .iter()
                .map(|placement| (placement.center - root).length())
                .sum();
            sum / children.len() as f32
        };
        let base = distance(&baseline);
        assert!(base > 0.1, "expected outward copies to start away from the parent");
        assert!((distance(&further) / base - 2.0).abs() < 1e-4);
        assert!((distance(&closer) / base - 0.5).abs() < 1e-4);

        params.depth = 2;
        params.gap = -GAP_LIMIT;
        let together = arrange(&params, 512, 512);
        assert!(distance(&together) < 1e-4);
        for placement in &together.placements {
            assert!(
                (placement.center - root).length() < 1e-3,
                "depth {} stayed {:.3} away",
                placement.depth,
                (placement.center - root).length()
            );
        }
    }

    #[test]
    fn corner_radius_rounds_every_layer_including_reflected_ones() {
        let clockwise = fillet_corner(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, -1.0),
            1.0,
            200.0,
        );
        assert!(!clockwise.arc.is_empty(), "clockwise corners should fillet");
        assert!((clockwise.arrive.x - 0.55).abs() < 1e-3);
        assert!(clockwise.arrive.y.abs() < 1e-3);
        assert!((clockwise.leave.y + 0.45).abs() < 1e-3);

        let mut params = FractalParams::default();
        params.depth = 2;
        params.shrink = 0.5;
        params.inward = false;
        params.corner_radius = 1.0;
        let rounded = generate(&params, 1024, 1024);
        let count = |depth| {
            rounded
                .segments
                .iter()
                .filter(|segment| segment.depth == depth)
                .count()
        };
        assert!(count(0) > 3, "root layer stayed sharp ({})", count(0));
        assert!(count(1) > 9, "middle layer stayed sharp ({})", count(1));
        assert!(count(2) > 18, "outer layer stayed sharp ({})", count(2));
    }

    #[test]
    fn stroke_opacity_defaults_stay_opaque_and_sanitize() {
        let params = FractalParams::default();
        assert_eq!(params.stroke_opacity, 1.0);
        assert_eq!(params.opacity_fade, 0.0);
        let clean = params.sanitized();
        assert_eq!(clean.stroke_opacity, 1.0);
        assert_eq!(clean.opacity_fade, 0.0);

        let mut params = FractalParams::default();
        params.stroke_opacity = f32::NAN;
        params.opacity_fade = f32::INFINITY;
        let clean = params.sanitized();
        assert_eq!(clean.stroke_opacity, 1.0);
        assert_eq!(clean.opacity_fade, 0.0);

        params.stroke_opacity = 4.0;
        params.opacity_fade = -0.5;
        let clean = params.sanitized();
        assert_eq!(clean.stroke_opacity, 1.0);
        assert_eq!(clean.opacity_fade, 0.0);

        params.stroke_opacity = -0.2;
        params.opacity_fade = 3.0;
        let clean = params.sanitized();
        assert_eq!(clean.stroke_opacity, 0.0);
        assert_eq!(clean.opacity_fade, 1.0);
    }

    #[test]
    fn stroke_taper_defaults_to_uniform_width_and_sanitizes() {
        let params = FractalParams::default();
        assert_eq!(params.stroke_taper, 0.0);
        assert_eq!(params.sanitized().stroke_taper, 0.0);

        let mut params = FractalParams::default();
        params.stroke_taper = f32::NAN;
        assert_eq!(params.sanitized().stroke_taper, 0.0);
        params.stroke_taper = f32::INFINITY;
        assert_eq!(params.sanitized().stroke_taper, 0.0);
        params.stroke_taper = 4.0;
        assert_eq!(params.sanitized().stroke_taper, 1.0);
        params.stroke_taper = -0.5;
        assert_eq!(params.sanitized().stroke_taper, 0.0);
    }

    #[test]
    fn randomize_is_sanitized_and_seed_dependent() {
        let mut first = FractalParams::default();
        let mut second = FractalParams::default();
        first.randomize(0x1234_5678);
        second.randomize(0x9ABC_DEF0);
        assert_eq!(first, first.sanitized());
        assert_eq!(second, second.sanitized());
        assert_ne!(first, second);
        assert!((3..=16).contains(&first.side_lengths.len()));
        assert!((1..=12).contains(&first.depth));
        assert!(first.palette.len() >= 2);
        assert!((0.0..=1.0).contains(&first.stroke_opacity));
        assert!((0.0..=1.0).contains(&first.stroke_taper));
        assert!((0.0..=0.5).contains(&first.jitter));

        let mut again = FractalParams::default();
        again.randomize(0x1234_5678);
        assert_eq!(first, again);

        let mut arranged = ArrangementParams::default();
        arranged.motif_by_depth = vec![7, 8, 9];
        arranged.depth = 2;
        let mut other = arranged.clone();
        arranged.randomize(11);
        other.randomize(22);
        assert_eq!(arranged, arranged.sanitized());
        assert_eq!(other, other.sanitized());
        assert_ne!(arranged, other);
        assert_eq!(
            arranged.motif_by_depth.len(),
            arranged.depth as usize + 1
        );
        let kept = arranged.motif_by_depth.len().min(3);
        assert_eq!(&arranged.motif_by_depth[..kept], &[7, 8, 9][..kept]);
        if arranged.motif_by_depth.len() > 3 {
            assert!(arranged.motif_by_depth[3..].iter().all(|id| *id == 9));
        }
    }

    #[test]
    fn background_gradient_defaults_to_solid_and_sanitizes_angle() {
        let params = FractalParams::default();
        assert_eq!(params.bg_mode, BgMode::Solid);
        assert_eq!(params.bg_end, [48, 72, 120]);
        assert_eq!(params.bg_angle, 0.0);
        assert_eq!(params.sanitized().bg_mode, BgMode::Solid);
        assert_eq!(params.sanitized().bg_angle, 0.0);

        let mut params = FractalParams::default();
        params.bg_mode = BgMode::Linear;
        params.bg_angle = f32::NAN;
        let clean = params.sanitized();
        assert_eq!(clean.bg_mode, BgMode::Linear);
        assert_eq!(clean.bg_angle, 0.0);

        let arrangement = ArrangementParams::default();
        assert_eq!(arrangement.bg_mode, BgMode::Solid);
        assert_eq!(arrangement.bg_end, [48, 72, 120]);
        assert_eq!(arrangement.bg_angle, 0.0);
        let mut arrangement = arrangement;
        arrangement.bg_mode = BgMode::Radial;
        arrangement.bg_angle = f32::INFINITY;
        let clean = arrangement.sanitized();
        assert_eq!(clean.bg_mode, BgMode::Radial);
        assert_eq!(clean.bg_angle, 0.0);
    }

    #[test]
    fn symmetry_one_keeps_the_segment_count_and_four_multiplies_it() {
        let defaults = FractalParams::default();
        assert_eq!(defaults.symmetry, 1);
        assert!(!defaults.mirror);
        assert_eq!(defaults.sanitized().symmetry, 1);
        assert!(!defaults.sanitized().mirror);

        let mut params = FractalParams::default();
        params.depth = 2;
        params.symmetry = 1;
        params.mirror = false;
        let once = generate(&params, 512, 512);
        assert_eq!(once.segments.len(), 30);
        assert!(!once.truncated);

        params.symmetry = 4;
        let four = generate(&params, 512, 512);
        assert_eq!(four.segments.len(), once.segments.len() * 4);
        assert!(!four.truncated);
        for (original, kept) in once.segments.iter().zip(&four.segments) {
            assert_eq!(original.x0, kept.x0);
            assert_eq!(original.y0, kept.y0);
            assert_eq!(original.x1, kept.x1);
            assert_eq!(original.y1, kept.y1);
        }
        let source = once.segments[0];
        let turned = rotate_around(
            Vec2::new(source.x0, source.y0),
            Vec2::ZERO,
            std::f32::consts::FRAC_PI_2,
        );
        assert!(four.segments.iter().any(|segment| {
            (segment.x0 - turned.x).abs() < 1e-4 && (segment.y0 - turned.y).abs() < 1e-4
        }));

        params.symmetry = 0;
        assert_eq!(params.sanitized().symmetry, 1);
        params.symmetry = 99;
        assert_eq!(params.sanitized().symmetry, 24);

        let mut arrangement = ArrangementParams::default();
        assert_eq!(arrangement.symmetry, 1);
        assert!(!arrangement.mirror);
        arrangement.depth = 1;
        arrangement.symmetry = 1;
        let stamps = arrange(&arrangement, 256, 256);
        arrangement.symmetry = 4;
        let folded = arrange(&arrangement, 256, 256);
        assert_eq!(folded.placements.len(), stamps.placements.len() * 4);
        assert!(!folded.truncated);
        arrangement.symmetry = 40;
        assert_eq!(arrangement.sanitized().symmetry, 24);
    }

    #[test]
    fn glow_defaults_off_and_sanitizes() {
        let params = FractalParams::default();
        assert_eq!(params.glow, 0.0);
        assert_eq!(params.glow_radius, 8.0);
        assert_eq!(params.sanitized().glow, 0.0);
        assert_eq!(params.sanitized().glow_radius, 8.0);

        let mut params = params;
        params.glow = f32::NAN;
        params.glow_radius = 1000.0;
        let clean = params.sanitized();
        assert_eq!(clean.glow, 0.0);
        assert_eq!(clean.glow_radius, 64.0);

        params.glow = 9.0;
        params.glow_radius = -4.0;
        let clean = params.sanitized();
        assert_eq!(clean.glow, 4.0);
        assert_eq!(clean.glow_radius, 0.0);

        let arrangement = ArrangementParams::default();
        assert_eq!(arrangement.glow, 0.0);
        assert_eq!(arrangement.glow_radius, 8.0);
        let mut arrangement = arrangement;
        arrangement.glow = f32::INFINITY;
        arrangement.glow_radius = f32::NAN;
        let clean = arrangement.sanitized();
        assert_eq!(clean.glow, 0.0);
        assert_eq!(clean.glow_radius, 8.0);
    }

    fn legacy_reach(seed: &SeedShape, shrink: f32, depth: u32) -> f32 {
        let radius = seed
            .vertices
            .iter()
            .map(|vertex| vertex.length())
            .fold(0.0_f32, f32::max)
            .max(1e-3);
        if depth == 0 {
            return radius;
        }
        let mid = (seed.vertices[0] + seed.vertices[1]) * 0.5;
        let joint_radius = seed
            .vertices
            .iter()
            .map(|vertex| (*vertex - mid).length())
            .fold(0.0_f32, f32::max)
            .max(1e-3);
        let extra = if (shrink - 1.0).abs() < 1e-4 {
            joint_radius * depth as f32
        } else {
            let s = shrink;
            joint_radius * s * (1.0 - s.powi(depth as i32)) / (1.0 - s)
        };
        (radius + extra).max(1e-3)
    }

    fn mean_edge(output: &Output, depth: u16) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for segment in &output.segments {
            if segment.depth == depth {
                sum += (segment.x1 - segment.x0).hypot(segment.y1 - segment.y0);
                count += 1;
            }
        }
        sum / count as f32
    }

    #[test]
    fn empty_shrink_by_depth_matches_uniform_and_override_stays_in_bounds() {
        let defaults = FractalParams::default();
        assert!(defaults.shrink_by_depth.is_empty());
        assert!(defaults.sanitized().shrink_by_depth.is_empty());
        assert!(ArrangementParams::default().shrink_by_depth.is_empty());

        let mut dirty = FractalParams::default();
        dirty.shrink = 0.4;
        dirty.shrink_by_depth = vec![f32::NAN, -1.0, 0.5, 2.0];
        assert_eq!(
            dirty.sanitized().shrink_by_depth,
            vec![0.4, 0.05, 0.5, 0.95]
        );
        let mut arranged = ArrangementParams::default();
        arranged.shrink = 0.4;
        arranged.shrink_by_depth = vec![f32::INFINITY, 0.01, 0.7];
        assert_eq!(
            arranged.sanitized().shrink_by_depth,
            vec![0.4, 0.05, 0.7]
        );
        assert_eq!(shrink_at(0.5, &[], 1), 0.5);
        assert_eq!(shrink_at(0.5, &[0.2], 1), 0.5);
        assert_eq!(shrink_at(0.5, &[0.2, 0.8], 1), 0.8);

        let seed = close_seed(&[1.0, 1.3, 0.7, 1.1]);
        for shrink in [0.05_f32, 0.5, 0.95, 1.0] {
            for depth in [0_u32, 1, 6, 12] {
                assert_eq!(
                    conservative_reach(&seed, shrink, &[], depth),
                    legacy_reach(&seed, shrink, depth)
                );
            }
        }

        let mut params = FractalParams::default();
        params.side_lengths = vec![1.0, 1.3, 0.7, 1.1];
        params.depth = 3;
        params.shrink = 0.55;
        params.rotation_deg = 17.0;
        params.twist_deg = -11.0;
        params.corner_radius = 0.4;
        let empty = generate(&params, 640, 480);
        let mut uniform = params.clone();
        uniform.shrink_by_depth = vec![params.shrink; params.depth as usize + 1];
        let same = generate(&uniform, 640, 480);
        assert_eq!(empty.segments.len(), same.segments.len());
        assert_eq!(empty.bounds.min_x, same.bounds.min_x);
        assert_eq!(empty.bounds.min_y, same.bounds.min_y);
        assert_eq!(empty.bounds.max_x, same.bounds.max_x);
        assert_eq!(empty.bounds.max_y, same.bounds.max_y);
        for (left, right) in empty.segments.iter().zip(&same.segments) {
            assert_eq!(left.x0, right.x0);
            assert_eq!(left.y0, right.y0);
            assert_eq!(left.x1, right.x1);
            assert_eq!(left.y1, right.y1);
            assert_eq!(left.depth, right.depth);
        }

        let mut triangle = FractalParams::default();
        triangle.depth = 2;
        triangle.shrink = 0.5;
        triangle.corner_radius = 0.0;
        let classic = generate(&triangle, 512, 512);
        assert_eq!(classic.segments.len(), 30);
        let classic_bounds = classic.bounds;

        let mut varied = triangle.clone();
        varied.shrink_by_depth = vec![0.5, 0.8, 0.25];
        let changed = generate(&varied, 1024, 1024);
        assert_eq!(changed.segments.len(), classic.segments.len());
        let root = mean_edge(&changed, 0);
        assert!((mean_edge(&classic, 1) / mean_edge(&classic, 0) - 0.5).abs() < 1e-4);
        assert!((mean_edge(&changed, 1) / root - 0.8).abs() < 1e-4);
        assert!((mean_edge(&changed, 2) / root - 0.2).abs() < 1e-4);
        assert_ne!(changed.bounds.max_x, classic_bounds.max_x);

        let varied_seed = close_seed(&varied.side_lengths);
        let reach = conservative_reach(
            &varied_seed,
            varied.shrink,
            &varied.shrink_by_depth,
            varied.depth,
        );
        for segment in &changed.segments {
            assert!(segment.x0 >= changed.bounds.min_x && segment.x0 <= changed.bounds.max_x);
            assert!(segment.y0 >= changed.bounds.min_y && segment.y0 <= changed.bounds.max_y);
            assert!(segment.x1 >= changed.bounds.min_x && segment.x1 <= changed.bounds.max_x);
            assert!(segment.y1 >= changed.bounds.min_y && segment.y1 <= changed.bounds.max_y);
            assert!(segment.x0.hypot(segment.y0) <= reach + 1e-3);
            assert!(segment.x1.hypot(segment.y1) <= reach + 1e-3);
        }

        let mut wide = FractalParams::default();
        wide.side_lengths = vec![1.0, 1.5, 0.8, 1.2];
        wide.depth = 3;
        wide.shrink = 0.2;
        wide.rotation_deg = 12.0;
        wide.twist_deg = 15.0;
        wide.corner_radius = 0.0;
        wide.shrink_by_depth = vec![0.2, 0.9, 0.85, 0.8];
        let output = generate(&wide, 1024, 1024);
        let wide_seed = close_seed(&wide.side_lengths);
        let wide_reach = conservative_reach(
            &wide_seed,
            wide.shrink,
            &wide.shrink_by_depth,
            wide.depth,
        );
        let stale = legacy_reach(&wide_seed, wide.shrink, wide.depth);
        let mut farthest = 0.0_f32;
        for segment in &output.segments {
            farthest = farthest.max(segment.x0.hypot(segment.y0));
            farthest = farthest.max(segment.x1.hypot(segment.y1));
            assert!(segment.x0.hypot(segment.y0) <= wide_reach + 1e-3);
            assert!(segment.x1.hypot(segment.y1) <= wide_reach + 1e-3);
        }
        assert!(farthest > stale, "old series under-bounds a larger per-depth shrink");

        let mut arrangement = ArrangementParams::default();
        arrangement.depth = 2;
        arrangement.shrink = 0.5;
        let uniform_stamps = arrange(&arrangement, 512, 512);
        arrangement.shrink_by_depth = vec![0.5, 0.8, 0.25];
        let placed = arrange(&arrangement, 512, 512);
        assert_eq!(placed.placements.len(), uniform_stamps.placements.len());
        for placement in &placed.placements {
            let expected = match placement.depth {
                0 => 1.0,
                1 => 0.8,
                2 => 0.2,
                other => panic!("unexpected depth {other}"),
            };
            assert!(
                (placement.scale - expected).abs() < 1e-5,
                "depth {} scale {}",
                placement.depth,
                placement.scale
            );
        }
        for placement in &uniform_stamps.placements {
            let expected = match placement.depth {
                0 => 1.0,
                1 => 0.5,
                2 => 0.25,
                other => panic!("unexpected depth {other}"),
            };
            assert!((placement.scale - expected).abs() < 1e-5);
        }
    }

    fn assert_same_segments(left: &Output, right: &Output) {
        assert_eq!(left.segments.len(), right.segments.len());
        for (a, b) in left.segments.iter().zip(&right.segments) {
            assert_eq!(a.x0, b.x0);
            assert_eq!(a.y0, b.y0);
            assert_eq!(a.x1, b.x1);
            assert_eq!(a.y1, b.y1);
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.color, b.color);
            assert_eq!(a.batch, b.batch);
        }
    }

    fn positions_differ(left: &Output, right: &Output) -> bool {
        left.segments.len() != right.segments.len()
            || left.segments.iter().zip(&right.segments).any(|(a, b)| {
                a.x0 != b.x0 || a.y0 != b.y0 || a.x1 != b.x1 || a.y1 != b.y1
            })
    }

    #[test]
    fn vertex_jitter_zero_matches_plain_geometry_and_positive_jitter_is_stable() {
        let defaults = FractalParams::default();
        assert_eq!(defaults.jitter, 0.0);
        assert_eq!(defaults.jitter_seed, 1);
        assert_eq!(defaults.sanitized().jitter, 0.0);
        assert_eq!(defaults.sanitized().jitter_seed, 1);

        let mut dirty = FractalParams::default();
        dirty.jitter = f32::NAN;
        assert_eq!(dirty.sanitized().jitter, 0.0);
        dirty.jitter = f32::INFINITY;
        assert_eq!(dirty.sanitized().jitter, 0.0);
        dirty.jitter = -0.2;
        assert_eq!(dirty.sanitized().jitter, 0.0);
        dirty.jitter = 1.25;
        assert_eq!(dirty.sanitized().jitter, 0.5);

        let mut params = FractalParams::default();
        params.depth = 2;
        params.shrink = 0.5;
        params.corner_radius = 0.0;
        params.jitter = 0.0;
        params.jitter_seed = 1;
        let baseline = generate(&params, 512, 512);
        assert_eq!(baseline.segments.len(), 30);

        let seed = close_seed(&params.side_lengths);
        let root = root_vertices(&seed.vertices, params.rotation_deg.to_radians());
        let root_segs: Vec<_> = baseline.segments.iter().filter(|segment| segment.depth == 0).collect();
        assert_eq!(root_segs.len(), root.len());
        for (index, segment) in root_segs.iter().enumerate() {
            let start = root[index];
            let end = root[(index + 1) % root.len()];
            assert_eq!(segment.x0, start.x);
            assert_eq!(segment.y0, start.y);
            assert_eq!(segment.x1, end.x);
            assert_eq!(segment.y1, end.y);
        }

        params.jitter_seed = 0xDEAD_BEEF;
        let other_seed = generate(&params, 512, 512);
        assert_same_segments(&baseline, &other_seed);

        params.jitter = 0.2;
        params.jitter_seed = 7;
        let once = generate(&params, 512, 512);
        let twice = generate(&params, 512, 512);
        assert_same_segments(&once, &twice);
        assert!(
            positions_differ(&baseline, &once),
            "jitter should move vertices"
        );

        let root_once: Vec<_> = once.segments.iter().filter(|segment| segment.depth == 0).collect();
        assert_eq!(root_once.len(), root_segs.len());
        let root_limit = params.jitter * seed.max_edge + 1e-4;
        let mut root_moved = false;
        for (before, after) in root_segs.iter().zip(&root_once) {
            let shift = (before.x0 - after.x0)
                .hypot(before.y0 - after.y0)
                .max((before.x1 - after.x1).hypot(before.y1 - after.y1));
            assert!(shift <= root_limit, "root shift {shift} exceeded {root_limit}");
            if shift > 1e-4 {
                root_moved = true;
            }
        }
        assert!(root_moved, "root vertices should move when jitter > 0");

        let jittered_root: Vec<Vec2> = root_once
            .iter()
            .map(|segment| Vec2::new(segment.x0, segment.y0))
            .collect();
        for (index, segment) in root_once.iter().enumerate() {
            let next = jittered_root[(index + 1) % jittered_root.len()];
            assert!((segment.x1 - next.x).abs() < 1e-5);
            assert!((segment.y1 - next.y).abs() < 1e-5);
        }
        let placement = placement_for_edge(
            &seed.vertices,
            params.shrink,
            jittered_root[0],
            jittered_root[1],
            centroid(&jittered_root),
            params.inward,
            params.twist_deg.to_radians(),
            1,
        );
        let expected = instance_vertices(&seed.vertices, placement);
        let child = &once.segments[root_once.len()..root_once.len() + expected.len()];
        assert!(child.iter().all(|segment| segment.depth == 1));
        let child_limit = params.jitter * seed.max_edge * placement.scale + 1e-4;
        let mut child_moved = false;
        for (index, segment) in child.iter().enumerate() {
            let start = expected[index];
            let end = expected[(index + 1) % expected.len()];
            let shift = (segment.x0 - start.x)
                .hypot(segment.y0 - start.y)
                .max((segment.x1 - end.x).hypot(segment.y1 - end.y));
            assert!(
                shift <= child_limit,
                "child shift {shift} exceeded {child_limit}"
            );
            if shift > 1e-4 {
                child_moved = true;
            }
        }
        assert!(child_moved, "child vertices should leave the parent edge");

        params.jitter_seed = 8;
        let other = generate(&params, 512, 512);
        assert!(
            positions_differ(&once, &other),
            "a different jitter seed should change the figure"
        );
        let repeated = generate(&params, 512, 512);
        assert_same_segments(&other, &repeated);
    }

    #[test]
    fn fill_none_skips_outlines_and_solid_follows_symmetry() {
        let defaults = FractalParams::default();
        assert_eq!(defaults.fill_mode, FillMode::None);
        assert_eq!(defaults.fill_opacity, 1.0);
        assert_eq!(defaults.sanitized().fill_mode, FillMode::None);
        assert_eq!(defaults.sanitized().fill_opacity, 1.0);

        let mut params = defaults;
        params.fill_opacity = f32::NAN;
        assert_eq!(params.sanitized().fill_opacity, 1.0);
        params.fill_opacity = 3.0;
        assert_eq!(params.sanitized().fill_opacity, 1.0);
        params.fill_opacity = -0.2;
        assert_eq!(params.sanitized().fill_opacity, 0.0);

        params = FractalParams::default();
        params.depth = 2;
        params.fill_mode = FillMode::None;
        let open = generate(&params, 256, 256);
        assert!(open.outlines.is_empty());
        assert_eq!(open.segments.len(), 30);

        params.fill_mode = FillMode::Solid;
        assert_eq!(fill_color(&params, 0), params.color_start);
        let filled = generate(&params, 256, 256);
        assert_eq!(filled.outlines.len(), 10);
        assert!(filled.outlines.iter().all(|outline| outline.verts.len() >= 3));
        assert_eq!(open.segments.len(), filled.segments.len());
        for (before, after) in open.segments.iter().zip(&filled.segments) {
            assert_eq!(before.x0, after.x0);
            assert_eq!(before.y0, after.y0);
            assert_eq!(before.x1, after.x1);
            assert_eq!(before.y1, after.y1);
            assert_eq!(before.depth, after.depth);
            assert_eq!(before.color, after.color);
        }

        params.palette = vec![[0, 255, 0], [0, 0, 255]];
        params.fill_mode = FillMode::Palette;
        params.gradient_hsv = false;
        assert_eq!(fill_color(&params, 0), [0, 255, 0]);
        assert_eq!(fill_color(&params, params.depth), [0, 0, 255]);

        params.fill_mode = FillMode::Solid;
        params.symmetry = 4;
        let four = generate(&params, 256, 256);
        assert_eq!(four.outlines.len(), filled.outlines.len() * 4);
        let source = filled.outlines[0].verts[0];
        let turned = rotate_around(source, Vec2::ZERO, std::f32::consts::FRAC_PI_2);
        assert!(four.outlines.iter().any(|outline| {
            outline.verts.iter().any(|point| {
                (point.x - turned.x).abs() < 1e-4 && (point.y - turned.y).abs() < 1e-4
            })
        }));

        params.symmetry = 1;
        params.mirror = true;
        let mirrored = generate(&params, 256, 256);
        assert_eq!(mirrored.outlines.len(), filled.outlines.len() * 2);
        let reflected = reflect_across(source, Vec2::ZERO, Vec2::new(1.0, 0.0));
        assert!(mirrored.outlines.iter().any(|outline| {
            outline.verts.iter().any(|point| {
                (point.x - reflected.x).abs() < 1e-4 && (point.y - reflected.y).abs() < 1e-4
            })
        }));

        params.mirror = false;
        params.depth = 1;
        params.corner_radius = 1.0;
        let rounded = generate(&params, 1024, 1024);
        let root = rounded
            .outlines
            .iter()
            .find(|outline| outline.depth == 0)
            .expect("root outline");
        assert!(
            root.verts.len() > 3,
            "fillet samples belong on the outline, got {}",
            root.verts.len()
        );
    }
}

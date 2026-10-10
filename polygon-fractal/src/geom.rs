//! Closed-polygon edge fractal.
//!
//! The seed is a polygon whose side lengths are walked with equal exterior
//! turns, then closed by spreading the leftover gap around the outline.
//! Each free edge grows a uniformly scaled copy of that seed.

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Hard stop so a high side count and depth cannot allocate without bound.
const MAX_SEGMENTS: usize = 1_500_000;

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

/// How each outline is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourMode {
    /// Every line uses `color_start`.
    Single,
    /// Blend `color_start` into `color_end` by depth.
    Gradient,
    /// A fresh colour for each line, from `color_seed`.
    Random,
}

/// Parameters edited by the control panel.
#[derive(Clone, Debug, PartialEq)]
pub struct FractalParams {
    pub side_lengths: Vec<f32>,
    pub shrink: f32,
    pub depth: u32,
    pub rotation_deg: f32,
    pub inward: bool,
    pub twist_deg: f32,
    pub colour_mode: ColourMode,
    pub color_start: [u8; 3],
    pub color_end: [u8; 3],
    /// Seed for [`ColourMode::Random`]. A new value gives a new set of colours.
    pub color_seed: u64,
    pub background: [u8; 3],
    /// Stroke width as a fraction of the mean target side length.
    pub stroke_fraction: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            side_lengths: vec![1.0, 1.0, 1.0],
            shrink: 0.5,
            depth: 6,
            rotation_deg: 0.0,
            inward: false,
            twist_deg: 0.0,
            colour_mode: ColourMode::Gradient,
            color_start: [255, 200, 0],
            color_end: [165, 242, 243],
            color_seed: 1,
            background: [0, 0, 0],
            stroke_fraction: 0.02,
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
        params
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

pub struct Output {
    pub segments: Vec<Segment>,
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
    let reach = conservative_reach(&seed, params.shrink, params.depth);
    let pixels_per_world = cull_pixels_per_world(width.max(1), height.max(1), reach);
    let min_world = 0.5 / pixels_per_world.max(1e-6);

    let root = root_vertices(&seed.vertices, params.rotation_deg.to_radians());
    let mut segments = Vec::new();
    push_outline(&root, 0, &mut segments);

    let budget = AtomicUsize::new(segments.len());
    let truncated = AtomicBool::new(false);
    let twist = params.twist_deg.to_radians();

    if params.depth >= 1 && seed.max_edge * params.shrink >= min_world {
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
                    params.shrink,
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

        let parts: Vec<Vec<Segment>> = jobs
            .into_par_iter()
            .map(|job| grow_tree(&seed, &params, job, min_world, twist, &budget, &truncated))
            .collect();
        for part in parts {
            segments.extend(part);
        }
    }

    paint_segments(&mut segments, &params);
    let stroke_world = params.stroke_fraction * seed.mean_target;
    let bounds = Bounds::from_segments(&segments);
    Output {
        segments,
        bounds,
        stroke_world,
        truncated: truncated.load(Ordering::Relaxed),
    }
}

fn grow_tree(
    seed: &SeedShape,
    params: &FractalParams,
    root_job: Job,
    min_world: f32,
    twist: f32,
    budget: &AtomicUsize,
    truncated: &AtomicBool,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut stack = vec![root_job];
    while let Some(job) = stack.pop() {
        let verts = instance_vertices(&seed.vertices, job.placement);
        let n = verts.len();
        let used = budget.fetch_add(n, Ordering::Relaxed);
        if used + n > MAX_SEGMENTS {
            truncated.store(true, Ordering::Relaxed);
            break;
        }
        push_outline(&verts, job.depth, &mut segments);
        if job.depth >= params.depth {
            continue;
        }
        let next_scale = job.placement.scale * params.shrink;
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
    segments
}

fn push_outline(verts: &[Vec2], depth: u32, out: &mut Vec<Segment>) {
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        out.push(Segment {
            x0: a.x,
            y0: a.y,
            x1: b.x,
            y1: b.y,
            depth: depth as u16,
            color: [0, 0, 0],
        });
    }
}

fn conservative_reach(seed: &SeedShape, shrink: f32, depth: u32) -> f32 {
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

fn cull_pixels_per_world(width: u32, height: u32, reach: f32) -> f32 {
    let avail = (width.min(height) as f32) * (1.0 - 2.0 * MARGIN);
    avail / (2.0 * reach)
}

fn paint_segments(segments: &mut [Segment], params: &FractalParams) {
    let mut rng = ColourRng(params.color_seed | 1);
    for segment in segments {
        segment.color = match params.colour_mode {
            ColourMode::Single => params.color_start,
            ColourMode::Gradient => depth_color(params, segment.depth as u32),
            ColourMode::Random => rng.next_colour(),
        };
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
}

pub fn depth_color(params: &FractalParams, depth: u32) -> [u8; 3] {
    let max_depth = params.depth.max(1) as f32;
    let t = (depth as f32 / max_depth).clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    [
        mix(params.color_start[0], params.color_end[0]),
        mix(params.color_start[1], params.color_end[1]),
        mix(params.color_start[2], params.color_end[2]),
    ]
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
}

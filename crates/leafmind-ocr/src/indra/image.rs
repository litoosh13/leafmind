//! The image steps Indra's Python pipeline does with OpenCV: area and linear resizing, rotation, quarter turns
//! and connected regions, on one grey byte per pixel.

/// A grey image, one byte per pixel, row by row.
#[derive(Clone)]
pub(crate) struct Grey {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Grey {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }
}

/// Python's `round` (half to even), used wherever Indra rounds a size.
pub(crate) fn round_even(v: f64) -> usize {
    v.round_ties_even().max(0.0) as usize
}

/// Per output pixel, the source pixels and their weights.
type Taps = Vec<Vec<(usize, f32)>>;

/// OpenCV's INTER_AREA: each output pixel averages the source pixels it covers, by covered share.
fn area_taps(src: usize, dst: usize) -> Taps {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let (a, b) = (d as f64 * scale, (d + 1) as f64 * scale);
            (a.floor() as usize..(b.ceil() as usize).min(src))
                .filter_map(|s| {
                    let cover = b.min(s as f64 + 1.0) - a.max(s as f64);
                    (cover > 1e-9).then_some((s, (cover / scale) as f32))
                })
                .collect()
        })
        .collect()
}

/// OpenCV's INTER_LINEAR: pixel centres aligned, edges repeated.
fn linear_taps(src: usize, dst: usize) -> Taps {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5).max(0.0);
            let s = (f.floor() as usize).min(src - 1);
            let t = (f - s as f64) as f32;
            let s1 = (s + 1).min(src - 1);
            if t <= 0.0 || s1 == s {
                vec![(s, 1.0)]
            } else {
                vec![(s, 1.0 - t), (s1, t)]
            }
        })
        .collect()
}

/// `img` resized to `w` × `h`: by area when `area`, else linearly.
pub(crate) fn resize(img: &Grey, w: usize, h: usize, area: bool) -> Grey {
    if (w, h) == (img.w, img.h) {
        return img.clone();
    }
    let taps = |src, dst| {
        if area && dst < src {
            area_taps(src, dst)
        } else {
            linear_taps(src, dst)
        }
    };
    let (tx, ty) = (taps(img.w, w), taps(img.h, h));
    let mut rows = vec![0f32; img.h * w];
    for y in 0..img.h {
        let src = &img.px[y * img.w..(y + 1) * img.w];
        for (x, t) in tx.iter().enumerate() {
            rows[y * w + x] = t.iter().map(|&(s, k)| src[s] as f32 * k).sum();
        }
    }
    let mut px = vec![0u8; w * h];
    for (y, t) in ty.iter().enumerate() {
        for x in 0..w {
            let v: f32 = t.iter().map(|&(s, k)| rows[s * w + x] * k).sum();
            px[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Grey { w, h, px }
}

/// OpenCV's getRotationMatrix2D about the image centre, scale 1: `degrees` counter-clockwise on screen.
fn rotation(degrees: f64, w: usize, h: usize) -> [f64; 6] {
    let (b, a) = degrees.to_radians().sin_cos();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    [
        a,
        b,
        (1.0 - a) * cx - b * cy,
        -b,
        a,
        b * cx + (1.0 - a) * cy,
    ]
}

/// `img` turned `degrees` counter-clockwise about its centre, same size, bilinear (OpenCV's warpAffine).
/// Outside the image: `fill`, or the nearest edge pixel when `None`.
pub(crate) fn rotate(img: &Grey, degrees: f64, fill: Option<u8>) -> Grey {
    let [a, b, tx, _, _, ty] = rotation(degrees, img.w, img.h);
    let mut px = vec![0u8; img.w * img.h];
    let (wm, hm) = (img.w as f64 - 1.0, img.h as f64 - 1.0);
    for y in 0..img.h {
        // The inverse of [[a, b], [-b, a]] is [[a, -b], [b, a]]; along a row the source moves by (a, b).
        let dy = y as f64 - ty;
        let (mut sx, mut sy) = (-a * tx - b * dy, -b * tx + a * dy);
        for x in 0..img.w {
            if x > 0 {
                sx += a;
                sy += b;
            }
            // Well inside: the four neighbours directly.
            if sx >= 0.0 && sy >= 0.0 && sx < wm && sy < hm {
                let (x0, y0) = (sx as usize, sy as usize);
                let (fx, fy) = ((sx - x0 as f64) as f32, (sy - y0 as f64) as f32);
                let i = y0 * img.w + x0;
                let p = |k: usize| img.px[k] as f32;
                let top = p(i) + (p(i + 1) - p(i)) * fx;
                let bottom = p(i + img.w) + (p(i + img.w + 1) - p(i + img.w)) * fx;
                px[y * img.w + x] = (top + (bottom - top) * fy).round() as u8;
                continue;
            }
            let inside = sx >= -1.0 && sy >= -1.0 && sx <= wm + 1.0 && sy <= hm + 1.0;
            px[y * img.w + x] = match fill {
                Some(f) if !inside => f,
                _ => {
                    let sample = |xx: f64, yy: f64| -> f32 {
                        if fill.is_some() && (xx < 0.0 || yy < 0.0 || xx > wm || yy > hm) {
                            fill.unwrap_or(0) as f32
                        } else {
                            img.at(xx.clamp(0.0, wm) as usize, yy.clamp(0.0, hm) as usize) as f32
                        }
                    };
                    let (x0, y0) = (sx.floor(), sy.floor());
                    let (fx, fy) = ((sx - x0) as f32, (sy - y0) as f32);
                    let top = sample(x0, y0) * (1.0 - fx) + sample(x0 + 1.0, y0) * fx;
                    let bottom =
                        sample(x0, y0 + 1.0) * (1.0 - fx) + sample(x0 + 1.0, y0 + 1.0) * fx;
                    (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8
                }
            };
        }
    }
    Grey {
        w: img.w,
        h: img.h,
        px,
    }
}

/// A box of an image made by `rotate(.., degrees, ..)` (size `w` × `h`), mapped back onto the image before it:
/// the bounding box of the four mapped corners, clipped.
pub(crate) fn unrotate_box(b: [f32; 4], degrees: f64, w: usize, h: usize) -> [f32; 4] {
    let [a, bb, tx, c, d, ty] = rotation(-degrees, w, h);
    let corners = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])];
    let pts: Vec<(f64, f64)> = corners
        .iter()
        .map(|&(x, y)| {
            let (x, y) = (x as f64, y as f64);
            (a * x + bb * y + tx, c * x + d * y + ty)
        })
        .collect();
    let fold = |f: fn(f64, f64) -> f64, i: usize, init: f64| {
        pts.iter()
            .map(|p| if i == 0 { p.0 } else { p.1 })
            .fold(init, f)
    };
    [
        fold(f64::min, 0, f64::MAX).clamp(0.0, w as f64) as f32,
        fold(f64::min, 1, f64::MAX).clamp(0.0, h as f64) as f32,
        fold(f64::max, 0, f64::MIN).clamp(0.0, w as f64) as f32,
        fold(f64::max, 1, f64::MIN).clamp(0.0, h as f64) as f32,
    ]
}

/// `img` turned a quarter turn clockwise (numpy's `rot90(img, -1)`).
pub(crate) fn turn_clockwise(img: &Grey) -> Grey {
    let (w, h) = (img.h, img.w);
    let mut px = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = img.at(y, img.h - 1 - x);
        }
    }
    Grey { w, h, px }
}

/// A box on an image turned a quarter turn clockwise, mapped back onto the image before it (height `h`).
pub(crate) fn unturn_box(b: [f32; 4], h: usize) -> [f32; 4] {
    let h = h as f32;
    [b[1], h - b[2], b[3], h - b[0]]
}

/// The 4-connected regions of `mask` (`w` wide), in raster order of their first pixel: each region's pixel
/// indices.
pub(crate) fn regions(mask: &[bool], w: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut out = vec![];
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let (mut region, mut stack) = (vec![], vec![start]);
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if i + w < mask.len() {
                visit(i + w);
            }
        }
        out.push(region);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_and_turns_keep_the_picture() {
        let img = Grey {
            w: 4,
            h: 2,
            px: vec![0, 0, 255, 255, 0, 0, 255, 255],
        };
        // Halving by area averages pairs.
        assert_eq!(resize(&img, 2, 1, true).px, vec![0, 255]);
        // A quarter turn clockwise: the left column becomes the top row.
        let t = turn_clockwise(&img);
        assert_eq!((t.w, t.h), (2, 4));
        assert_eq!(t.px, vec![0, 0, 0, 0, 255, 255, 255, 255]);
        // A box on the turned image maps back onto the original.
        assert_eq!(unturn_box([0.0, 2.0, 2.0, 4.0], 2), [2.0, 0.0, 4.0, 2.0]);
        // No rotation leaves the image as it is.
        assert_eq!(rotate(&img, 0.0, Some(0)).px, img.px);
        let b = unrotate_box([1.0, 0.5, 3.0, 1.5], 0.0, 4, 2);
        assert!((b[0] - 1.0).abs() < 1e-4 && (b[3] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn regions_are_four_connected() {
        // 1 . 1
        // 1 . .
        // . 1 1   (diagonal neighbours do not connect)
        let mask = [true, false, true, true, false, false, false, true, true];
        let mut found = regions(&mask, 3);
        found.iter_mut().for_each(|r| r.sort());
        assert_eq!(found, vec![vec![0, 3], vec![2], vec![7, 8]]);
    }
}

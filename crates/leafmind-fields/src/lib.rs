//! Finds form fields on a page image: text boxes, choice buttons (checkboxes and radio buttons) and
//! signature areas. Uses Nutrient's `form-field-v1-nano` detector (Apache-2.0, see THIRD_PARTY.md),
//! run by tract, so it works natively and in the browser with no other runtime. Needs no image library
//! at run time: the page scaling is our own port of Pillow's bilinear resize. Optionally Nutrient's
//! `form-field-v1-state` classifier (Apache-2.0, 0.4 MB) tells whether each field is already filled.
//!
//! ```no_run
//! let finder = leafmind_fields::FieldFinder::from_onnx(&std::fs::read("models/form-field-v1-nano.onnx")?)?
//!     .with_state(&std::fs::read("models/form-field-v1-state.onnx")?)?; // optional
//! # let (rgba, width, height) = (vec![255u8; 4 * 100 * 100], 100, 100);
//! for field in finder.find(&rgba, width, height)? {
//!     println!("{:?} at {:?} ({:.2}), filled: {:?}", field.kind, field.bounds, field.score, field.filled);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;
use std::sync::Arc;
use tract_onnx::prelude::*;

#[cfg(feature = "wasm")]
mod wasm;

/// Side of the square image the model looks at.
const SIZE: u32 = 640;
/// Grey the model expects around the page (letterbox padding).
const PAD: f32 = 114.0;
/// The state model's input: a field crop letterboxed to 40 × 160 pixels (fields are wide), padded white.
const STATE_H: usize = 40;
const STATE_W: usize = 160;

/// What a found field is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// Somewhere to write text.
    Text,
    /// A checkbox or radio button.
    Choice,
    /// Somewhere to sign.
    Signature,
}

impl FieldKind {
    fn from_class(class: usize) -> Self {
        match class {
            0 => Self::Text,
            1 => Self::Choice,
            _ => Self::Signature,
        }
    }
}

/// A field found on the page.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub kind: FieldKind,
    /// The model's confidence, 0 to 1.
    pub score: f32,
    /// Left, top, right, bottom in pixels of the image passed to [`FieldFinder::find`].
    pub bounds: [f32; 4],
    /// Whether the field is already filled (text written, box ticked, signed); `None` without the state
    /// model ([`FieldFinder::with_state`]).
    pub filled: Option<bool>,
}

/// Tuning for [`FieldFinder::find_with`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Fields below this confidence are dropped.
    pub min_score: f32,
    /// Of two same-kind fields overlapping more than this (intersection over union), the weaker is dropped.
    pub max_overlap: f32,
}

impl Default for Options {
    /// 0.2 and 0.6. The model card says 0.3; on scans and filled forms that drops many real fields (blind
    /// test of public forms: 64 % found on filled scans at 0.3, 78 % at 0.2; forms with real fields 80 % →
    /// 88 %), for somewhat more wrong boxes.
    fn default() -> Self {
        Self {
            min_score: 0.2,
            max_overlap: 0.6,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    /// The model file could not be loaded.
    Model(String),
    /// The image buffer does not match its width and height.
    Image(String),
    /// Running the model failed.
    Run(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(m) => write!(f, "could not load the model: {m}"),
            Self::Image(m) => write!(f, "bad image: {m}"),
            Self::Run(m) => write!(f, "could not run the model: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// The loaded detector. Load once, then call [`find`](Self::find) for each page.
pub struct FieldFinder {
    plan: Arc<TypedSimplePlan>,
    state: Option<Arc<TypedSimplePlan>>,
}

impl FieldFinder {
    /// Loads the detector from the bytes of `form-field-v1-nano.onnx`.
    pub fn from_onnx(model: &[u8]) -> Result<Self, Error> {
        let err = |e: TractError| Error::Model(e.to_string());
        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model))
            .map_err(err)?
            .with_input_fact(0, f32::fact([1, 3, SIZE as usize, SIZE as usize]).into())
            .map_err(err)?
            .into_optimized()
            .map_err(err)?
            .into_runnable()
            .map_err(err)?;
        Ok(Self { plan, state: None })
    }

    /// Adds Nutrient's `form-field-v1-state` classifier (the bytes of its `model.onnx`): then every field
    /// [`find`](Self::find) returns says whether it is filled.
    pub fn with_state(mut self, model: &[u8]) -> Result<Self, Error> {
        let err = |e: TractError| Error::Model(e.to_string());
        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model))
            .map_err(err)?
            .with_input_fact(0, f32::fact([1, 1, STATE_H, STATE_W]).into())
            .map_err(err)?
            .into_optimized()
            .map_err(err)?
            .into_runnable()
            .map_err(err)?;
        self.state = Some(plan);
        Ok(self)
    }

    /// Finds the fields on a page image (RGBA, 8 bits per channel, `width` × `height` pixels),
    /// with the default [`Options`].
    pub fn find(&self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<Field>, Error> {
        self.find_with(rgba, width, height, Options::default())
    }

    /// Like [`find`](Self::find), with your own [`Options`].
    pub fn find_with(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        options: Options,
    ) -> Result<Vec<Field>, Error> {
        if width == 0 || height == 0 || rgba.len() != width as usize * height as usize * 4 {
            return Err(Error::Image(format!(
                "{} bytes for {width}×{height} RGBA",
                rgba.len()
            )));
        }
        let (input, scale) = letterbox(rgba, width, height);
        let mut fields = self.run(input, scale, options)?;
        if let Some(state) = &self.state {
            for field in &mut fields {
                field.filled = Some(filled(state, rgba, width, height, &field.bounds)?);
            }
        }
        Ok(fields)
    }

    /// Runs the model on an image that is already letterboxed to 640×640 (RGB, 8 bits, row by row),
    /// `scale` being the page-to-640 factor used. For tests against other implementations of the
    /// letterbox step.
    #[doc(hidden)]
    pub fn find_prepared(
        &self,
        rgb: &[u8],
        scale: f32,
        options: Options,
    ) -> Result<Vec<Field>, Error> {
        let plane = (SIZE * SIZE) as usize;
        if rgb.len() != plane * 3 {
            return Err(Error::Image(format!("{} bytes for 640×640 RGB", rgb.len())));
        }
        let mut input = vec![0.0; 3 * plane];
        for (i, pixel) in rgb.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                input[c * plane + i] = pixel[c] as f32;
            }
        }
        self.run(input, scale, options)
    }

    fn run(&self, input: Vec<f32>, scale: f32, options: Options) -> Result<Vec<Field>, Error> {
        let tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, SIZE as usize, SIZE as usize), input)
                .map_err(|e| Error::Run(e.to_string()))?
                .into_tensor();
        let outputs = self
            .plan
            .run(tvec!(tensor.into()))
            .map_err(|e| Error::Run(e.to_string()))?;
        let rows = outputs[0]
            .cast_to::<f32>()
            .map_err(|e| Error::Run(e.to_string()))?;
        let rows = rows
            .to_plain_array_view::<f32>()
            .map_err(|e| Error::Run(e.to_string()))?;
        let rows: Vec<f32> = rows.iter().copied().collect();
        Ok(decode(&rows, scale, options))
    }
}

/// Scales the page to fit 640×640 (top-left aligned, grey padding) as the model's reference recipe does,
/// into CHW floats 0–255. The scaling is a faithful port of Pillow's `resize(…, BILINEAR)` (see
/// [`bilinear`]), so results match the recipe the model was benchmarked with.
fn letterbox(rgba: &[u8], width: u32, height: u32) -> (Vec<f32>, f32) {
    let scale = (SIZE as f32 / width as f32).min(SIZE as f32 / height as f32);
    // Python computes the target size in double precision: int(width * r) with r = 640 / max side.
    let r = (SIZE as f64 / width as f64).min(SIZE as f64 / height as f64);
    let (w, h) = (
        ((width as f64 * r) as usize).max(1),
        ((height as f64 * r) as usize).max(1),
    );
    let small = bilinear(rgba, width as usize, height as usize, w, h);
    let plane = (SIZE * SIZE) as usize;
    let mut input = vec![PAD; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let (i, j) = (y * SIZE as usize + x, (y * w + x) * 4);
            input[i] = small[j] as f32;
            input[plane + i] = small[j + 1] as f32;
            input[2 * plane + i] = small[j + 2] as f32;
        }
    }
    (input, scale)
}

/// Runs the state model on one field, as its model card does: the crop in grey (Pillow's "L"), scaled
/// keeping its shape to fit 40 × 160 (Pillow's bilinear resize, Python's rounding), centred on white, then
/// `(x / 255 − 0.5) / 0.5`. Filled when the second logit is the larger.
fn filled(
    plan: &Arc<TypedSimplePlan>,
    rgba: &[u8],
    width: u32,
    height: u32,
    bounds: &[f32; 4],
) -> Result<bool, Error> {
    let (w, h) = (width as usize, height as usize);
    let x0 = (bounds[0].max(0.0) as usize).min(w - 1);
    let y0 = (bounds[1].max(0.0) as usize).min(h - 1);
    let x1 = (bounds[2].ceil().max(0.0) as usize).clamp(x0 + 1, w);
    let y1 = (bounds[3].ceil().max(0.0) as usize).clamp(y0 + 1, h);
    let (cw, ch) = (x1 - x0, y1 - y0);
    // Grey as Pillow's RGB → L (ITU-R 601-2 luma, 16-bit fixed point), kept in RGBA for the resize.
    let mut grey = Vec::with_capacity(cw * ch * 4);
    for y in y0..y1 {
        for p in rgba[(y * w + x0) * 4..(y * w + x1) * 4].as_chunks::<4>().0 {
            let l = ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000)
                >> 16) as u8;
            grey.extend([l, l, l, 255]);
        }
    }
    let s = (STATE_W as f64 / cw as f64).min(STATE_H as f64 / ch as f64);
    let nw = ((cw as f64 * s).round_ties_even() as usize).clamp(1, STATE_W);
    let nh = ((ch as f64 * s).round_ties_even() as usize).clamp(1, STATE_H);
    let small = bilinear(&grey, cw, ch, nw, nh);
    let (ox, oy) = ((STATE_W - nw) / 2, (STATE_H - nh) / 2);
    let mut input = vec![1.0f32; STATE_H * STATE_W]; // white: (255 / 255 − 0.5) / 0.5
    for y in 0..nh {
        for x in 0..nw {
            input[(oy + y) * STATE_W + ox + x] = small[(y * nw + x) * 4] as f32 / 127.5 - 1.0;
        }
    }
    let tensor = tract_ndarray::Array4::from_shape_vec((1, 1, STATE_H, STATE_W), input)
        .map_err(|e| Error::Run(e.to_string()))?
        .into_tensor();
    let outputs = plan
        .run(tvec!(tensor.into()))
        .map_err(|e| Error::Run(e.to_string()))?;
    let logits = outputs[0]
        .to_plain_array_view::<f32>()
        .map_err(|e| Error::Run(e.to_string()))?;
    let logits: Vec<f32> = logits.iter().copied().collect();
    Ok(logits[1] > logits[0])
}

/// Fixed-point precision of Pillow's 8-bit resampling (libImaging/Resample.c: 32 - 8 - 2).
const PRECISION_BITS: u32 = 22;

/// Pillow's convolution resize with the bilinear (triangle) filter, 8 bits per channel, RGBA in and out:
/// per output pixel, weights over the source pixels within `support` (1 × the scale factor when
/// shrinking), normalised, turned into 22-bit fixed point, horizontal pass then vertical pass, each
/// rounded and clipped to 8 bits. Mirrors `precompute_coeffs`, `normalize_coeffs_8bpc` and
/// `ImagingResampleHorizontal/Vertical_8bpc`.
fn bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let (xb, xk, xn) = coefficients(sw, dw);
    let (yb, yk, yn) = coefficients(sh, dh);
    let first = yb[0].0;
    let last = yb[dh - 1].0 + yb[dh - 1].1;
    let rows = last - first;
    let half = 1i32 << (PRECISION_BITS - 1);
    let clip = |v: i32| -> u8 {
        if v <= 0 {
            0
        } else if v >= 1 << (PRECISION_BITS + 8) {
            255
        } else {
            (v >> PRECISION_BITS) as u8
        }
    };
    let mut tmp = vec![0u8; dw * rows * 4];
    for y in 0..rows {
        let line = &src[(y + first) * sw * 4..];
        for x in 0..dw {
            let (min, n) = xb[x];
            let k = &xk[x * xn..x * xn + n];
            for c in 0..4 {
                let mut acc = half;
                for (i, w) in k.iter().enumerate() {
                    acc += line[(min + i) * 4 + c] as i32 * w;
                }
                tmp[(y * dw + x) * 4 + c] = clip(acc);
            }
        }
    }
    let mut out = vec![0u8; dw * dh * 4];
    for y in 0..dh {
        let (min, n) = yb[y];
        let min = min - first;
        let k = &yk[y * yn..y * yn + n];
        for x in 0..dw {
            for c in 0..4 {
                let mut acc = half;
                for (i, w) in k.iter().enumerate() {
                    acc += tmp[((min + i) * dw + x) * 4 + c] as i32 * w;
                }
                out[(y * dw + x) * 4 + c] = clip(acc);
            }
        }
    }
    out
}

/// For each output index: (first source index, count) and fixed-point weights (`stride` per output).
fn coefficients(input: usize, output: usize) -> (Vec<(usize, usize)>, Vec<i32>, usize) {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter_scale; // triangle filter support 1.0 × filter scale
    let stride = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(output);
    let mut weights = vec![0i32; output * stride];
    for xx in 0..output {
        let center = (xx as f64 + 0.5) * scale;
        let min = ((center - support + 0.5) as i64).max(0) as usize;
        let max = ((center + support + 0.5) as i64).min(input as i64) as usize - min;
        let mut k = vec![0f64; max];
        for (x, kx) in k.iter_mut().enumerate() {
            let t = ((x + min) as f64 - center + 0.5) / filter_scale;
            *kx = (1.0 - t.abs()).max(0.0);
        }
        let sum: f64 = k.iter().sum();
        for (x, kx) in k.iter().enumerate() {
            let w = if sum != 0.0 { kx / sum } else { *kx };
            let fixed = w * (1u32 << PRECISION_BITS) as f64;
            weights[xx * stride + x] = if w < 0.0 {
                (-0.5 + fixed) as i32
            } else {
                (0.5 + fixed) as i32
            };
        }
        bounds.push((min, max));
    }
    (bounds, weights, stride)
}

/// Model rows are `[cx, cy, w, h, objectness, p_text, p_choice, p_signature]` in 640-pixel space.
/// Score = objectness × class probability; keep the best class, drop weak boxes, then per kind drop
/// boxes overlapping a stronger one (non-maximum suppression); map back to page pixels.
fn decode(rows: &[f32], scale: f32, options: Options) -> Vec<Field> {
    let mut by_kind: [Vec<Field>; 3] = Default::default();
    for row in rows.as_chunks::<8>().0 {
        let (class, score) = (0..3)
            .map(|k| (k, row[4] * row[5 + k]))
            .fold((0, f32::MIN), |best, c| if c.1 > best.1 { c } else { best });
        if score < options.min_score {
            continue;
        }
        let (cx, cy, w, h) = (row[0], row[1], row[2], row[3]);
        let bounds = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0].map(|v| v / scale);
        by_kind[class].push(Field {
            kind: FieldKind::from_class(class),
            score,
            bounds,
            filled: None,
        });
    }
    let mut fields = Vec::new();
    for mut list in by_kind {
        list.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Field> = Vec::new();
        for field in list {
            if kept
                .iter()
                .all(|k| overlap(&k.bounds, &field.bounds) <= options.max_overlap)
            {
                kept.push(field);
            }
        }
        fields.extend(kept);
    }
    // The model sometimes reads one spot as two kinds (a weak "signature" over a "text" box): a field mostly
    // covered by a stronger field of another kind goes. Blind test of public forms at threshold 0.2: wrong
    // boxes 111 → 89, fields found 1,668 → 1,666 of 1,906.
    let covered = |a: &Field, b: &Field| {
        let w = (a.bounds[2].min(b.bounds[2]) - a.bounds[0].max(b.bounds[0])).max(0.0);
        let h = (a.bounds[3].min(b.bounds[3]) - a.bounds[1].max(b.bounds[1])).max(0.0);
        w * h / ((a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1])).max(1e-6)
    };
    let weak: Vec<bool> = fields
        .iter()
        .map(|f| {
            fields
                .iter()
                .any(|g| g.kind != f.kind && g.score > f.score && covered(f, g) > 0.5)
        })
        .collect();
    fields
        .into_iter()
        .zip(weak)
        .filter_map(|(f, weak)| (!weak).then_some(f))
        .collect()
}

/// Intersection over union of two boxes.
pub fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
}

#[cfg(test)]
mod tests {
    /// Our letterbox must equal Pillow's pixel for pixel (tests/data/*.model-input.png were made with
    /// Pillow 12's `resize(…, BILINEAR)` by tests/data/make_testdata.py).
    #[test]
    fn letterbox_matches_pillow_exactly() {
        let root = env!("CARGO_MANIFEST_DIR");
        for page in ["library-card"] {
            let img = image::open(format!("{root}/tests/data/{page}.png"))
                .unwrap()
                .to_rgba8();
            let (w, h) = img.dimensions();
            let (ours, _) = super::letterbox(img.as_raw(), w, h);
            let pillow = image::open(format!("{root}/tests/data/{page}.model-input.png"))
                .unwrap()
                .to_rgb8();
            let plane = 640 * 640;
            let mut differ = 0;
            for (i, p) in pillow.pixels().enumerate() {
                for c in 0..3 {
                    differ += (ours[c * plane + i] != p[c] as f32) as usize;
                }
            }
            assert_eq!(
                differ, 0,
                "{page}: {differ} channel values differ from Pillow"
            );
        }
    }

    #[test]
    fn a_weaker_field_of_another_kind_on_the_same_spot_goes() {
        // Rows as the model writes them: centre x, centre y, width, height, objectness, three class scores.
        let rows = [
            100.0, 50.0, 200.0, 20.0, 0.9, 0.9, 0.0, 0.0, // text, 0.81
            100.0, 52.0, 180.0, 18.0, 0.9, 0.0, 0.0,
            0.3, // signature on the same spot, 0.27: goes
            100.0, 150.0, 180.0, 18.0, 0.9, 0.0, 0.0, 0.3, // signature elsewhere, 0.27: stays
        ];
        let kinds: Vec<_> = super::decode(&rows, 1.0, super::Options::default())
            .iter()
            .map(|f| (f.kind, f.bounds[1]))
            .collect();
        assert_eq!(
            kinds,
            [
                (super::FieldKind::Text, 40.0),
                (super::FieldKind::Signature, 141.0)
            ]
        );
    }
}

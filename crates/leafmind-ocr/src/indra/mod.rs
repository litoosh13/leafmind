//! Indra OCR (Datalab) on ONNX Runtime, as an optional engine next to Tesseract (feature `indra`). Reads every
//! script it knows on one page, without choosing a language first, and is several times faster than Tesseract.
//!
//! A port of the indra-ocr 1.1.2 pipeline (Apache-2.0, github.com/datalab-to/indra): page straightened and turned
//! upright (orient.onnx), text lines found (det.onnx, a second pass for small text), each line's script
//! (script.onnx) sends it to the Latin recognizer (rec.onnx) or a script group's (rec_rtl.onnx: Arabic,
//! Persian, Hebrew …; rec_world.onnx; rec_cjk.onnx), right-to-left lines put in logical order, reading order by
//! XY-cut, paragraphs by line spacing. Left out, measured on leafmind's scan set (no change in character errors or
//! answers): the n-gram language model for unsure lines and the reading-order / layout model.
//!
//! **The model weights are not open source**: a modified OpenRAIL-M licence (free for research, personal use and
//! organisations under a revenue / funding limit; not for products competing with Datalab's). leafmind ships no
//! weights; an app that uses this engine downloads them (huggingface.co/datalab-to/indra-ocr) and must accept
//! their licence. Tesseract ([`crate::OcrEngine`]) stays the default.

mod image;
mod lines;
mod text;

use crate::{Error, OcrPage, OcrWord, grey};
use image::{Grey, regions, resize, rotate, round_even};
use lines::{Decoder, Line, Tok};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use text::{Vocab, is_rtl};

/// Where the app keeps ONNX Runtime and Indra's model folder.
#[derive(Clone, Debug)]
pub struct IndraModels {
    /// The ONNX Runtime library (the same one leafmind-qa uses).
    pub onnxruntime: PathBuf,
    /// Folder with det.onnx, rec.onnx, meta.json and, optionally, orient.onnx, script.onnx and rec_<group>.onnx.
    pub models: PathBuf,
}

/// Line height (detector input pixels) under which the page is detected again, larger.
const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f32 = 12.0;
const ADAPTIVE_MAX_SIDE: usize = 2048;
/// A turn is applied only above this probability (orient.onnx): real pages are almost all upright.
const ORIENT_MIN: f32 = 0.95;
const ORIENT_SIDE: usize = 768;
const PREVIEW_SIDE: usize = 1600;
const DESKEW_MAX: f64 = 45.0;
const DESKEW_MIN: f64 = 0.3;
/// A line read under this confidence gets no text (empty fields, rules and checkboxes read as "-", "I", "1").
const MIN_CONF: f32 = 0.5;
/// A line goes to one recognizer when the script classifier is surer than this; else the Latin one reads it too.
const ROUTE_SURE: f32 = 0.95;
/// Cut a line at a gap of this many line heights (between character centres).
const SPLIT_GAP: f32 = 3.0;
const GUTTER_GAP: f32 = 1.6;
/// A new paragraph after a gap of this many median line heights. Measured on leafmind's scan Q&A set: 0.5, 0.8
/// and 1.2 answer 69–71 questions, as Indra's own layout blocks (70).
const PARAGRAPH_GAP: f32 = 0.8;
/// Line crops: padding in line heights, and the recognizers' input height.
const PAD_X: f32 = 0.3;
const PAD_Y: f32 = 0.2;
const WIDTH_MULTIPLE: usize = 8;

/// One ONNX model; running it needs `&mut`, hence the lock.
struct Net(Mutex<Session>);

fn model_error(e: impl std::fmt::Display) -> Error {
    Error::Model(e.to_string())
}

impl Net {
    fn load(path: &Path, threads: usize) -> Result<Net, Error> {
        let session = Session::builder()
            .map_err(model_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_error)?
            .with_intra_threads(threads)
            .map_err(model_error)?
            .with_inter_threads(1)
            .map_err(model_error)?
            // Line batches change shape on every call: a memory pattern or arena only grows.
            .with_memory_pattern(false)
            .map_err(model_error)?
            .commit_from_file(path)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        Ok(Net(Mutex::new(session)))
    }

    fn load_if(path: PathBuf, threads: usize) -> Result<Option<Net>, Error> {
        path.exists().then(|| Net::load(&path, threads)).transpose()
    }
}

/// A recognizer and its vocabulary.
struct Reader {
    net: Net,
    vocab: Vocab,
}

/// Loaded Indra models. Can be shared between threads (each model runs one call at a time).
pub struct IndraEngine {
    det: Net,
    orient: Option<Net>,
    script: Option<Net>,
    /// The Latin recognizer; it also reads lines the script classifier is unsure of.
    latin: Reader,
    /// Script groups in script.onnx's class order; `None` where the folder has no recognizer for it.
    groups: Vec<(String, Option<Reader>)>,
    long_side: usize,
    height: usize,
    /// The Latin recognizer reads a window this many line heights tall (the line and the page around it).
    window: usize,
    threads: usize,
}

/// One reading of a line by one recognizer.
struct Reading {
    toks: Vec<Tok>,
    conf: f32,
    chars: bool,
}

impl IndraEngine {
    /// Loads ONNX Runtime (once per process) and the models. `threads`: per model call; 0 = half the cores.
    pub fn load(models: &IndraModels, threads: usize) -> Result<Self, Error> {
        ort::init_from(&models.onnxruntime)
            .map_err(model_error)?
            .commit();
        let threads = match threads {
            0 => std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1)),
            n => n,
        };
        let dir = &models.models;
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json"))
                .map_err(|e| Error::Model(format!("meta.json: {e}")))?,
        )
        .map_err(model_error)?;
        let vocab = |key: &str| -> Result<Vocab, Error> {
            Vocab::from_json(meta[key].as_str().unwrap_or_default()).map_err(Error::Model)
        };
        let names: Vec<String> = match meta["script_groups"].as_array() {
            Some(a) => a
                .iter()
                .filter_map(|g| g.as_str().map(String::from))
                .collect(),
            None => ["base", "cjk", "rtl", "other"].map(String::from).to_vec(),
        };
        let mut groups = vec![];
        for name in names {
            let path = dir.join(format!("rec_{name}.onnx"));
            let key = format!("vocab_{name}");
            let reader = if name != "base" && path.exists() && meta.get(&key).is_some() {
                Some(Reader {
                    net: Net::load(&path, threads)?,
                    vocab: vocab(&key)?,
                })
            } else {
                None
            };
            groups.push((name, reader));
        }
        let has_groups = groups.iter().any(|g| g.1.is_some());
        Ok(IndraEngine {
            det: Net::load(&dir.join("det.onnx"), threads)?,
            orient: Net::load_if(dir.join("orient.onnx"), threads)?,
            script: if has_groups {
                Net::load_if(dir.join("script.onnx"), threads)?
            } else {
                None
            },
            latin: Reader {
                net: Net::load(&dir.join("rec.onnx"), threads)?,
                vocab: vocab("vocab")?,
            },
            groups,
            long_side: meta["det_long_side"].as_u64().unwrap_or(1024) as usize,
            height: meta["rec_height"].as_u64().unwrap_or(32) as usize,
            window: meta["rec_window"].as_u64().unwrap_or(1) as usize,
            threads,
        })
    }

    /// Reads a page image (RGBA). Boxes are in pixels of this image. `OcrPage::text` has one paragraph per block
    /// of lines, a blank line between; words carry their line's confidence.
    pub fn read(&self, rgba: &[u8], width: u32, height: u32) -> Result<OcrPage, Error> {
        let page = Grey {
            w: width as usize,
            h: height as usize,
            px: grey(rgba, width, height)?,
        };
        // Straightened and turned upright on a preview first; the orientation model reads it straightened.
        let small = preview(&page);
        let mut skew = self.estimate_skew(&small);
        let turns = self.orientation(&if skew.abs() >= DESKEW_MIN {
            rotate(&small, skew, None)
        } else {
            small.clone()
        })?;
        let (mut img, mut small) = (page.clone(), small);
        for _ in 0..turns {
            img = image::turn_clockwise(&img);
            small = image::turn_clockwise(&small);
        }
        if turns % 2 == 1 {
            // Rows were columns: measure the angle again.
            skew = self.estimate_skew(&small);
        }
        if skew.abs() >= DESKEW_MIN {
            img = rotate(&img, skew, None);
        } else {
            skew = 0.0;
        }
        let boxes = self.detect(&img)?;
        let readings = self.recognize(&img, &boxes)?;
        let mut out = page_text(&self.assemble(&img, boxes, readings));
        // Word boxes back onto the image as given.
        for word in &mut out.words {
            if skew != 0.0 {
                word.bounds = image::unrotate_box(word.bounds, skew, img.w, img.h);
            }
            for t in (0..turns).rev() {
                word.bounds =
                    image::unturn_box(word.bounds, if t % 2 == 0 { page.h } else { page.w });
            }
        }
        Ok(out)
    }

    /// Degrees (counter-clockwise) that make the text rows level: the angle whose row ink profile changes most
    /// sharply, searched in 2° steps on a small thumbnail, then 0.5° and 0.1° near the best ones.
    fn estimate_skew(&self, img: &Grey) -> f64 {
        let fine = ink_thumb(img, 800);
        if (fine.px.iter().map(|&v| v as f64).sum::<f64>() / fine.px.len().max(1) as f64)
            < 0.003 * 255.0
        {
            return 0.0;
        }
        let coarse = ink_thumb(img, 400);
        let grid: Vec<f64> = (0..=45).map(|i| -DESKEW_MAX + 2.0 * i as f64).collect();
        let scores = par_map(&grid, self.threads, |&a| row_sharpness(&coarse, a));
        let mut by_score: Vec<usize> = (0..grid.len()).collect();
        by_score.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        let starts = [
            0.0,
            grid[by_score[grid.len() - 1]],
            grid[by_score[grid.len() - 2]],
        ];
        let mut near: Vec<f64> = vec![];
        for a0 in starts {
            for i in 0..=8 {
                let a = ((a0 - 2.0 + 0.5 * i as f64) * 100.0).round() / 100.0;
                if !near.contains(&a) {
                    near.push(a);
                }
            }
        }
        let best = |angles: &[f64]| {
            let s = par_map(angles, self.threads, |&a| row_sharpness(&fine, a));
            let mut k = 0;
            for i in 1..angles.len() {
                if s[i] > s[k] {
                    k = i;
                }
            }
            angles[k]
        };
        let a1 = best(&near);
        let last: Vec<f64> = (0..=10)
            .map(|i| ((a1 - 0.5 + 0.1 * i as f64) * 100.0).round() / 100.0)
            .collect();
        best(&last)
    }

    /// Quarter turns clockwise that put the page upright (orient.onnx), when the model is sure.
    fn orientation(&self, img: &Grey) -> Result<usize, Error> {
        let Some(net) = &self.orient else {
            return Ok(0);
        };
        let s = ORIENT_SIDE as f64 / img.w.max(img.h) as f64;
        let small = resize(
            img,
            round_even(img.w as f64 * s).max(1),
            round_even(img.h as f64 * s).max(1),
            true,
        );
        let mut x = vec![0f32; ORIENT_SIDE * ORIENT_SIDE];
        for y in 0..small.h {
            for xx in 0..small.w {
                x[y * ORIENT_SIDE + xx] = 1.0 - small.at(xx, y) as f32 / 255.0;
            }
        }
        let mut session = net.0.lock().unwrap();
        let out = session
            .run(ort::inputs!["x" => Tensor::from_array(([1, 1, ORIENT_SIDE, ORIENT_SIDE], x)).map_err(model_error)?])
            .map_err(model_error)?;
        let (_, p) = out[0].try_extract_tensor::<f32>().map_err(model_error)?;
        let (k, &pk) = p
            .iter()
            .enumerate()
            .fold((0, &p[0]), |a, b| if b.1 > a.1 { b } else { a });
        // Class 4 (a page set in vertical columns, Chinese or Japanese) is read as it is.
        // ponytail: vertical CJK pages unsupported; add Indra's vertical path if such documents matter.
        Ok(if pk > ORIENT_MIN && k < 4 { k } else { 0 })
    }

    /// Text-line boxes [x0, y0, x1, y1] in page pixels.
    fn detect(&self, img: &Grey) -> Result<Vec<[f32; 4]>, Error> {
        let (mut boxes, mut s) = self.detect_at(img, self.long_side)?;
        if boxes.len() >= 8 {
            let med = lines_median(boxes.iter().map(|b| b[3] - b[1]).collect());
            if med < ADAPTIVE_MIN_H {
                let side = ((ADAPTIVE_MAX_SIDE as f32)
                    .min(self.long_side as f32 * ADAPTIVE_TARGET_H / med.max(1.0))
                    as usize)
                    / 32
                    * 32;
                if side > self.long_side {
                    (boxes, s) = self.detect_at(img, side)?;
                }
            }
        }
        let (w, h) = (img.w as f32, img.h as f32);
        Ok(boxes
            .into_iter()
            .map(|b| {
                [
                    (b[0] / s).clamp(0.0, w),
                    (b[1] / s).clamp(0.0, h),
                    (b[2] / s).clamp(0.0, w),
                    (b[3] / s).clamp(0.0, h),
                ]
            })
            .collect())
    }

    /// Boxes in detector input pixels at `long_side`, and the scale.
    fn detect_at(&self, img: &Grey, long_side: usize) -> Result<(Vec<[f32; 4]>, f32), Error> {
        let s = long_side as f64 / img.w.max(img.h) as f64;
        let (nw, nh) = (
            round_even(img.w as f64 * s).max(1),
            round_even(img.h as f64 * s).max(1),
        );
        let small = resize(img, nw, nh, s < 1.0);
        let (pw, ph) = (nw.div_ceil(32) * 32, nh.div_ceil(32) * 32);
        let mut x = vec![0f32; pw * ph];
        for y in 0..nh {
            for xx in 0..nw {
                x[y * pw + xx] = 1.0 - small.at(xx, y) as f32 / 255.0;
            }
        }
        let mut session = self.det.0.lock().unwrap();
        let out = session
            .run(ort::inputs!["x" => Tensor::from_array(([1, 1, ph, pw], x)).map_err(model_error)?])
            .map_err(model_error)?;
        let (shape, v) = out[0].try_extract_tensor::<f32>().map_err(model_error)?;
        let (oh, ow) = (shape[2] as usize, shape[3] as usize);
        let plane = oh * ow;
        let (core, top, bottom) = (&v[..plane], &v[plane..2 * plane], &v[2 * plane..3 * plane]);
        // Line cores above probability 0.5; each region's box from its mean top and bottom distances.
        let mask: Vec<bool> = core.iter().map(|&l| l > 0.0).collect();
        let mut boxes = vec![];
        for region in regions(&mask, ow) {
            if region.len() < 2 {
                continue;
            }
            let n = region.len() as f32;
            let (mut y0, mut y1, mut left, mut right) = (0.0, 0.0, usize::MAX, 0);
            for &i in &region {
                let ys = ((i / ow) as f32 + 0.5) * 2.0;
                y0 += ys - top[i];
                y1 += ys + bottom[i];
                left = left.min(i % ow);
                right = right.max(i % ow + 1);
            }
            let (y0, y1) = (y0 / n, y1 / n);
            let ex = 0.15 * (y1 - y0).max(1.0);
            boxes.push([left as f32 * 2.0 - ex, y0, right as f32 * 2.0 + ex, y1]);
        }
        Ok((boxes, s as f32))
    }

    /// Each line read by the recognizer of its script (the Latin one too when the classifier is unsure); the
    /// more confident reading wins.
    fn recognize(&self, img: &Grey, boxes: &[[f32; 4]]) -> Result<Vec<Option<Reading>>, Error> {
        let dark: Vec<bool> = boxes.iter().map(|b| is_dark_line(img, b)).collect();
        let band: Vec<Crop> = boxes
            .iter()
            .zip(&dark)
            .map(|(b, &d)| crop_line(img, b, self.height, 1, d))
            .collect();
        let tall: Vec<Crop> = if self.window > 1 {
            boxes
                .iter()
                .zip(&dark)
                .map(|(b, &d)| crop_line(img, b, self.height, self.window, d))
                .collect()
        } else {
            band.clone()
        };
        let (route, sure) = self.route(&band)?;
        let mut best: Vec<Option<Reading>> = (0..boxes.len()).map(|_| None).collect();
        let mut read =
            |reader: &Reader, crops: &[Crop], ids: Vec<usize>, chars: bool| -> Result<(), Error> {
                for part in width_batches(&ids.iter().map(|&j| band[j].w).collect::<Vec<_>>()) {
                    let idx: Vec<usize> = part.iter().map(|&i| ids[i]).collect();
                    let (x, widths, h, w) =
                        pad_batch(&idx.iter().map(|&j| &crops[j]).collect::<Vec<_>>());
                    let mut session = reader.net.0.lock().unwrap();
                    let out = session
                    .run(ort::inputs![
                        "x" => Tensor::from_array(([idx.len(), 1, h, w], x)).map_err(model_error)?,
                        "widths" => Tensor::from_array(([idx.len()], widths)).map_err(model_error)?,
                    ])
                    .map_err(model_error)?;
                    let (shape, seq) = out["ids"]
                        .try_extract_tensor::<i64>()
                        .map_err(model_error)?;
                    let (_, pmax) = out["conf"]
                        .try_extract_tensor::<f32>()
                        .map_err(model_error)?;
                    let (_, lengths) = out["lengths"]
                        .try_extract_tensor::<i64>()
                        .map_err(model_error)?;
                    let steps = shape[1] as usize;
                    let stride = w as f32 / steps.max(1) as f32;
                    for (k, &j) in idx.iter().enumerate() {
                        let n = (lengths[k].max(0) as usize).min(steps);
                        let (s, p) = (
                            &seq[k * steps..k * steps + n],
                            &pmax[k * steps..k * steps + n],
                        );
                        let b = boxes[j];
                        let h = (b[3] - b[1]).max(1.0);
                        let ix0 = (b[0] - PAD_X * h).floor();
                        let f = ((b[2] + PAD_X * h).ceil() - ix0) / band[j].w as f32;
                        let (mut toks, mut sum) = (vec![], 0.0);
                        for t in 0..n {
                            if s[t] != 0 && (t == 0 || s[t] != s[t - 1]) {
                                sum += p[t];
                                toks.push(Tok {
                                    text: reader.vocab.token(s[t]).to_string(),
                                    x: ix0 + (t as f32 + 0.5) * stride * f,
                                });
                            }
                        }
                        let conf = if toks.is_empty() {
                            0.0
                        } else {
                            sum / toks.len() as f32
                        };
                        if best[j].as_ref().is_none_or(|r| conf > r.conf) {
                            best[j] = Some(Reading { toks, conf, chars });
                        }
                    }
                }
                Ok(())
            };
        read(
            &self.latin,
            &tall,
            (0..boxes.len())
                .filter(|&j| route[j].is_none() || !sure[j])
                .collect(),
            false,
        )?;
        for (g, (_, reader)) in self.groups.iter().enumerate() {
            if let Some(reader) = reader {
                read(
                    reader,
                    &band,
                    (0..boxes.len()).filter(|&j| route[j] == Some(g)).collect(),
                    true,
                )?;
            }
        }
        Ok(best
            .into_iter()
            .map(|r| {
                r.map(|r| {
                    if r.conf >= MIN_CONF {
                        r
                    } else {
                        Reading { toks: vec![], ..r }
                    }
                })
            })
            .collect())
    }

    /// Per line, the script group whose recognizer reads it (`None`: the Latin one), and whether the classifier
    /// is sure.
    fn route(&self, crops: &[Crop]) -> Result<(Vec<Option<usize>>, Vec<bool>), Error> {
        let mut route = vec![None; crops.len()];
        let mut sure = vec![true; crops.len()];
        let Some(net) = &self.script else {
            return Ok((route, sure));
        };
        for idx in width_batches(&crops.iter().map(|c| c.w).collect::<Vec<_>>()) {
            let (x, widths, h, w) = pad_batch(&idx.iter().map(|&j| &crops[j]).collect::<Vec<_>>());
            let mut session = net.0.lock().unwrap();
            let out = session
                .run(ort::inputs![
                    "x" => Tensor::from_array(([idx.len(), 1, h, w], x)).map_err(model_error)?,
                    "widths" => Tensor::from_array(([idx.len()], widths)).map_err(model_error)?,
                ])
                .map_err(model_error)?;
            let (shape, p) = out[0].try_extract_tensor::<f32>().map_err(model_error)?;
            let classes = shape[1] as usize;
            for (k, &j) in idx.iter().enumerate() {
                let row = &p[k * classes..(k + 1) * classes];
                let (g, &pg) = row
                    .iter()
                    .enumerate()
                    .fold((0, &row[0]), |a, b| if b.1 > a.1 { b } else { a });
                route[j] = self.groups.get(g).filter(|(_, r)| r.is_some()).map(|_| g);
                sure[j] = pg > ROUTE_SURE;
            }
        }
        Ok((route, sure))
    }

    /// Lines cut at wide gaps and column gutters, leaders dropped, list markers joined, in XY-cut reading order
    /// (mirrored on a right-to-left page, so its right column comes first).
    fn assemble(
        &self,
        img: &Grey,
        boxes: Vec<[f32; 4]>,
        readings: Vec<Option<Reading>>,
    ) -> Vec<Line> {
        let raw: Vec<Line> = boxes
            .into_iter()
            .zip(readings)
            .filter_map(|(bounds, r)| {
                let r = r?;
                Some(Line {
                    bounds,
                    text: r.toks.iter().map(|t| t.text.as_str()).collect(),
                    toks: r.toks,
                    conf: r.conf,
                    chars: r.chars,
                })
            })
            .collect();
        let letters: Vec<char> = raw
            .iter()
            .flat_map(|l| l.text.chars())
            .filter(|c| c.is_alphabetic())
            .collect();
        let rtl_letters = letters.iter().filter(|&&c| is_rtl(c)).count();
        let page_rtl = rtl_letters > letters.len() - rtl_letters;
        let dec = Decoder {
            latin: &self.latin.vocab,
            page_rtl,
        };
        let cuts = lines::gutter_cuts(&raw, GUTTER_GAP);
        let split: Vec<Line> = raw
            .iter()
            .zip(&cuts)
            .flat_map(|(l, at)| lines::split(l, SPLIT_GAP, at, &dec))
            .collect();
        let tidy = lines::tidy(split, page_rtl);
        let w = img.w as f32;
        let geo: Vec<[f32; 4]> = tidy
            .iter()
            .map(|l| {
                let b = l.bounds;
                if page_rtl {
                    [w - b[2], b[1], w - b[0], b[3]]
                } else {
                    b
                }
            })
            .collect();
        lines::xycut(&geo)
            .into_iter()
            .map(|i| tidy[i].clone())
            .collect()
    }
}

/// The page's text (paragraphs by line spacing), median line confidence and words.
fn page_text(all: &[Line]) -> OcrPage {
    let lines: Vec<&Line> = all.iter().filter(|l| !l.text.trim().is_empty()).collect();
    let paras = lines::paragraphs(&lines, PARAGRAPH_GAP);
    let text = paras
        .iter()
        .map(|p| {
            p.iter()
                .map(|&i| lines[i].text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut words = vec![];
    for (n, l) in lines.iter().enumerate() {
        let h = (l.bounds[3] - l.bounds[1]).max(1.0);
        // Word boxes from the token positions where the line has them, else the line's words spread evenly.
        let groups: Vec<Vec<&Tok>> = l
            .toks
            .split(|t| t.text == " ")
            .filter(|g| !g.is_empty())
            .map(|g| g.iter().collect())
            .collect();
        let texts: Vec<&str> = l.text.split_whitespace().collect();
        if groups.len() == texts.len() && !l.chars {
            for (g, t) in groups.iter().zip(&texts) {
                let (lo, hi) = g
                    .iter()
                    .fold((f32::MAX, f32::MIN), |a, t| (a.0.min(t.x), a.1.max(t.x)));
                words.push(OcrWord {
                    text: t.to_string(),
                    bounds: [
                        l.bounds[0].max(lo - 0.35 * h),
                        l.bounds[1],
                        l.bounds[2].min(hi + 0.35 * h),
                        l.bounds[3],
                    ],
                    confidence: l.conf,
                    line: n as u32,
                });
            }
        } else {
            // ponytail: right-to-left and mixed lines get evenly spread word boxes; place them by token positions
            // if an app needs exact word boxes on those.
            let total: usize = texts
                .iter()
                .map(|t| t.chars().count() + 1)
                .sum::<usize>()
                .max(1);
            let span = l.bounds[2] - l.bounds[0];
            let mut at = 0usize;
            for t in &texts {
                let len = t.chars().count();
                let (a, b) = (at as f32 / total as f32, (at + len) as f32 / total as f32);
                at += len + 1;
                words.push(OcrWord {
                    text: t.to_string(),
                    bounds: [
                        l.bounds[0] + a * span,
                        l.bounds[1],
                        l.bounds[0] + b * span,
                        l.bounds[3],
                    ],
                    confidence: l.conf,
                    line: n as u32,
                });
            }
        }
    }
    OcrPage {
        text,
        confidence: lines_median(lines.iter().map(|l| l.conf).collect()),
        words,
    }
}

fn lines_median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The page shrunk once to at most `PREVIEW_SIDE` a side: the skew search and orientation read it.
fn preview(img: &Grey) -> Grey {
    let s = PREVIEW_SIDE as f64 / img.w.max(img.h) as f64;
    if s >= 1.0 {
        return img.clone();
    }
    resize(
        img,
        round_even(img.w as f64 * s).max(1),
        round_even(img.h as f64 * s).max(1),
        true,
    )
}

/// Ink map (255 = ink) of the page shrunk to `side`.
fn ink_thumb(img: &Grey, side: usize) -> Grey {
    let s = side as f64 / img.w.max(img.h) as f64;
    let mut t = resize(
        img,
        round_even(img.w as f64 * s).max(1),
        round_even(img.h as f64 * s).max(1),
        true,
    );
    t.px.iter_mut().for_each(|v| *v = 255 - *v);
    t
}

/// How sharply the row ink profile changes, the ink map turned by `a` degrees: highest when rows are level.
fn row_sharpness(ink: &Grey, a: f64) -> f64 {
    let r = rotate(ink, a, Some(0));
    let rows: Vec<f64> =
        r.px.chunks(r.w)
            .map(|row| row.iter().map(|&v| v as f64).sum())
            .collect();
    rows.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum()
}

/// `f` over `items` on up to `threads` threads, results in order.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let chunk = items.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
}

/// A recognizer input: an ink map (1 = ink), `h` rows of `w`.
#[derive(Clone)]
struct Crop {
    w: usize,
    h: usize,
    ink: Vec<f32>,
}

/// numpy's linear percentiles of the bytes in `b`, from their histogram.
fn percentiles(b: &[u8], qs: [f64; 3]) -> [f64; 3] {
    let n = b.len();
    let mut cum = [0usize; 256];
    for &v in b {
        cum[v as usize] += 1;
    }
    for i in 1..256 {
        cum[i] += cum[i - 1];
    }
    // The value at sorted position k: the first histogram bin whose running count passes k.
    let at = |k: usize| cum.iter().position(|&c| c > k).unwrap_or(255) as f64;
    qs.map(|q| {
        let x = q * (n - 1) as f64;
        let i = x.floor() as usize;
        let g = x - i as f64;
        let (lo, hi) = (at(i), at((i + 1).min(n - 1)));
        let d = hi - lo;
        if g >= 0.5 {
            hi - d * (1.0 - g)
        } else {
            lo + d * g
        }
    })
}

/// Light text on a dark ground: the ground (median) lies nearer the dark end than the text does. Such a line is
/// read inverted.
fn is_dark_line(img: &Grey, b: &[f32; 4]) -> bool {
    let (x0, y0) = (b[0].max(0.0) as usize, b[1].max(0.0) as usize);
    let (x1, y1) = (
        (b[2] as usize + 1).min(img.w),
        (b[3] as usize + 1).min(img.h),
    );
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let px: Vec<u8> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .map(|(x, y)| img.at(x, y))
        .collect();
    let [lo, med, hi] = percentiles(&px, [0.02, 0.5, 0.98]);
    hi - med > 2.0 * (med - lo) + 8.0
}

/// A line cropped with padding (white outside the page), scaled to `height` × `window` rows, as an ink map.
/// A tall window keeps the text scale and adds the page above and below.
fn crop_line(img: &Grey, b: &[f32; 4], height: usize, window: usize, dark: bool) -> Crop {
    let pad_y = (window as f32 * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let height = height * window;
    let h = (b[3] - b[1]).max(1.0);
    let (ix0, iy0) = (
        (b[0] - PAD_X * h).floor() as i64,
        (b[1] - pad_y * h).floor() as i64,
    );
    let (ix1, iy1) = (
        (b[2] + PAD_X * h).ceil() as i64,
        (b[3] + pad_y * h).ceil() as i64,
    );
    let (cw, ch) = ((ix1 - ix0).max(1) as usize, (iy1 - iy0).max(1) as usize);
    let paper = if dark { 0 } else { 255 };
    let mut px = vec![paper; cw * ch];
    for y in 0..ch {
        let sy = iy0 + y as i64;
        if sy < 0 || sy >= img.h as i64 {
            continue;
        }
        for x in 0..cw {
            let sx = ix0 + x as i64;
            if sx >= 0 && sx < img.w as i64 {
                px[y * cw + x] = img.at(sx as usize, sy as usize);
            }
        }
    }
    let out_w = round_even(cw as f64 * height as f64 / ch as f64).clamp(WIDTH_MULTIPLE, 4096);
    let r = resize(&Grey { w: cw, h: ch, px }, out_w, height, ch > height);
    let ink =
        r.px.iter()
            .map(|&v| {
                if dark {
                    v as f32 / 255.0
                } else {
                    1.0 - v as f32 / 255.0
                }
            })
            .collect();
    Crop {
        w: out_w,
        h: height,
        ink,
    }
}

/// Crops right-padded to one width (a multiple of 8): the batch, the widths, its height and width.
fn pad_batch(crops: &[&Crop]) -> (Vec<f32>, Vec<i64>, usize, usize) {
    let h = crops[0].h;
    let w = crops
        .iter()
        .map(|c| c.w)
        .max()
        .unwrap_or(1)
        .div_ceil(WIDTH_MULTIPLE)
        * WIDTH_MULTIPLE;
    let mut x = vec![0f32; crops.len() * h * w];
    for (k, c) in crops.iter().enumerate() {
        for y in 0..h {
            x[(k * h + y) * w..(k * h + y) * w + c.w]
                .copy_from_slice(&c.ink[y * c.w..(y + 1) * c.w]);
        }
    }
    (x, crops.iter().map(|c| c.w as i64).collect(), h, w)
}

/// Line indices in batches of similar width, narrowest first: a batch closes at 64 lines, at a line over 1.3
/// times as wide as its first, or past 64 × 640 pixel columns (ONNX Runtime keeps its largest allocation).
fn width_batches(widths: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]);
    let mut batches: Vec<Vec<usize>> = vec![];
    for i in order {
        let w = widths[i].max(64);
        match batches.last_mut() {
            Some(b)
                if b.len() < 64
                    && w as f64 <= 1.3 * widths[b[0]].max(64) as f64
                    && (b.len() + 1) * w <= 64 * 640 =>
            {
                b.push(i)
            }
            _ => batches.push(vec![i]),
        }
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_group_similar_widths() {
        // 40 counts as 64; 100 is over 1.3 × 64, so it starts the next batch.
        assert_eq!(
            width_batches(&[100, 300, 110, 40]),
            vec![vec![3], vec![0, 2], vec![1]]
        );
        let p = percentiles(&[0, 10, 20, 30, 40], [0.0, 0.5, 1.0]);
        assert_eq!(p, [0.0, 20.0, 40.0]);
    }
}

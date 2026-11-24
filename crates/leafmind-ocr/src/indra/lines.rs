//! Recognized lines to page text: lines cut at wide gaps, leader lines dropped and list markers joined to their
//! items (indra_ocr/pipeline.py), reading order by XY-cut (order_post.py), then paragraphs by line spacing.

use super::text::{Vocab, group_text};

/// A recognized token (a character, or a LaTeX command) and its x position on the page.
#[derive(Clone)]
pub(crate) struct Tok {
    pub text: String,
    pub x: f32,
}

impl Tok {
    fn is_space(&self) -> bool {
        self.text == " "
    }
}

/// One line: its box, tokens and confidence. `chars`: read by a script group's recognizer (visual order, so
/// right-to-left runs are turned round when decoded), else by the Latin recognizer.
#[derive(Clone)]
pub(crate) struct Line {
    pub bounds: [f32; 4],
    pub toks: Vec<Tok>,
    pub conf: f32,
    pub chars: bool,
    pub text: String,
}

/// Decodes tokens the way their recognizer wrote them.
pub(crate) struct Decoder<'a> {
    pub latin: &'a Vocab,
    pub page_rtl: bool,
}

impl Decoder<'_> {
    pub fn text(&self, toks: &[Tok], chars: bool) -> String {
        let s: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        if chars {
            group_text(&s.concat(), self.page_rtl)
        } else {
            self.latin.decode_tokens(&s)
        }
    }
}

/// Cuts a detected line where two neighbouring characters are over `gap` line heights apart, or at the
/// positions `at` (column gutters): the detector joined two cells or columns.
pub(crate) fn split(line: &Line, gap: f32, at: &[f32], dec: &Decoder) -> Vec<Line> {
    let b = line.bounds;
    let h = (b[3] - b[1]).max(1.0);
    let mut segs: Vec<Vec<Tok>> = vec![vec![]];
    let mut prev: Option<f32> = None;
    for t in &line.toks {
        if !t.is_space() {
            if let Some(p) = prev
                && (t.x - p > gap * h || at.iter().any(|&c| p < c && c < t.x))
            {
                segs.push(vec![]);
            }
            prev = Some(t.x);
        }
        segs.last_mut().unwrap().push(t.clone());
    }
    if segs.len() == 1 {
        return vec![Line {
            text: dec.text(&line.toks, line.chars),
            ..line.clone()
        }];
    }
    segs.into_iter()
        .filter_map(|sg| {
            let text = dec.text(&sg, line.chars).trim().to_string();
            let xs: Vec<f32> = sg.iter().filter(|t| !t.is_space()).map(|t| t.x).collect();
            if text.is_empty() || xs.is_empty() {
                return None;
            }
            let (lo, hi) = xs
                .iter()
                .fold((f32::MAX, f32::MIN), |a, &x| (a.0.min(x), a.1.max(x)));
            Some(Line {
                bounds: [b[0].max(lo - 0.35 * h), b[1], b[2].min(hi + 0.35 * h), b[3]],
                toks: sg,
                conf: line.conf,
                chars: line.chars,
                text,
            })
        })
        .collect()
}

/// Per line, the x positions of gaps on a column gutter: over `gutter_gap` line heights wide, with at least 3
/// other lines ending at its left edge or starting at its right edge.
pub(crate) fn gutter_cuts(lines: &[Line], gutter_gap: f32) -> Vec<Vec<f32>> {
    lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let h = (l.bounds[3] - l.bounds[1]).max(1.0);
            let xs: Vec<f32> = l
                .toks
                .iter()
                .filter(|t| !t.is_space())
                .map(|t| t.x)
                .collect();
            xs.windows(2)
                .filter(|w| w[1] - w[0] > gutter_gap * h)
                .filter(|w| {
                    let (gl, gr) = (w[0] + 0.35 * h, w[1] - 0.35 * h);
                    let near = lines
                        .iter()
                        .enumerate()
                        .filter(|&(j, o)| {
                            j != i
                                && ((o.bounds[2] - gl).abs() < 0.5 * h
                                    || (o.bounds[0] - gr).abs() < 0.5 * h)
                        })
                        .count();
                    near >= 3
                })
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect()
        })
        .collect()
}

const LEADERS: &str = ".·…‥_-–—=";
const DASHES: &str = "-–—";
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";
/// A list marker and its item's text are at most this many line heights apart.
const MARKER_GAP: f32 = 2.5;

/// A numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
fn enumerator(t: &str) -> bool {
    let num_or_letter = |s: &str| {
        (1..=3).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit())
            || s.len() == 1 && s.bytes().all(|b| b.is_ascii_alphabetic())
    };
    if let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        return num_or_letter(inner)
            || (1..=5).contains(&inner.len()) && inner.chars().all(|c| "ivxlcIVXLC".contains(c));
    }
    t.strip_suffix(['.', ')']).is_some_and(num_or_letter)
}

/// Drops lines of leader characters (dotted rules, specks; a short lone dash stays, it can be an empty table
/// cell) and joins a lone list marker to the item that starts on its row within `MARKER_GAP` line heights.
pub(crate) fn tidy(mut lines: Vec<Line>, rtl: bool) -> Vec<Line> {
    let mut keep = vec![true; lines.len()];
    for (i, l) in lines.iter().enumerate() {
        let core: String = l.text.chars().filter(|&c| c != ' ').collect();
        if !core.is_empty() && core.chars().all(|c| LEADERS.contains(c)) {
            let (w, h) = (l.bounds[2] - l.bounds[0], l.bounds[3] - l.bounds[1]);
            if core.chars().count() >= 3
                || w >= 3.0 * h
                || !core.chars().any(|c| DASHES.contains(c))
            {
                keep[i] = false;
            }
        }
    }
    for i in 0..lines.len() {
        let t = lines[i].text.trim().to_string();
        let marker = t.chars().count() == 1 && BULLETS.contains(t.as_str()) || enumerator(&t);
        if !keep[i] || !marker {
            continue;
        }
        let bi = lines[i].bounds;
        let hi = bi[3] - bi[1];
        let mut best: Option<(usize, f32)> = None;
        for (j, o) in lines.iter().enumerate() {
            if j == i || !keep[j] || o.text.trim().is_empty() {
                continue;
            }
            let bj = o.bounds;
            let hj = bj[3] - bj[1];
            if bi[3].min(bj[3]) - bi[1].max(bj[1]) < 0.5 * hi.min(hj) {
                continue;
            }
            let gap = if rtl { bi[0] - bj[2] } else { bj[0] - bi[2] };
            if -0.3 * hi <= gap && gap <= MARKER_GAP * hi.max(hj) && best.is_none_or(|b| gap < b.1)
            {
                best = Some((j, gap));
            }
        }
        let Some((j, _)) = best else { continue };
        let marker_line = lines[i].clone();
        let item = &mut lines[j];
        if !rtl
            && marker_line.chars == item.chars
            && !marker_line.toks.is_empty()
            && !item.toks.is_empty()
        {
            let x = (marker_line.bounds[2] + item.bounds[0]) / 2.0;
            let mut toks = marker_line.toks.clone();
            toks.push(Tok {
                text: " ".into(),
                x,
            });
            toks.append(&mut item.toks);
            item.toks = toks;
        }
        item.text = format!("{t} {}", item.text.trim());
        item.bounds = [
            bi[0].min(item.bounds[0]),
            bi[1].min(item.bounds[1]),
            bi[2].max(item.bounds[2]),
            bi[3].max(item.bounds[3]),
        ];
        keep[i] = false;
    }
    lines
        .into_iter()
        .zip(keep)
        .filter_map(|(l, k)| k.then_some(l))
        .collect()
}

fn median(mut v: Vec<f32>) -> f32 {
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

/// Rows top to bottom (centres within half a line height share a row), left to right.
fn top_down(boxes: &[[f32; 4]], ids: &[usize]) -> Vec<usize> {
    let h = median(ids.iter().map(|&i| boxes[i][3] - boxes[i][1]).collect());
    let row =
        |i: usize| (((boxes[i][1] + boxes[i][3]) / 2.0) / (h * 0.5).max(1e-3)).round_ties_even();
    let mut out = ids.to_vec();
    out.sort_by(|&a, &b| {
        row(a)
            .total_cmp(&row(b))
            .then(boxes[a][0].total_cmp(&boxes[b][0]))
    });
    out
}

/// Recursive XY-cut: split at the widest white gap on either axis, the upper or left part first; without a gap,
/// top-down order.
pub(crate) fn xycut(boxes: &[[f32; 4]]) -> Vec<usize> {
    fn best_cut(boxes: &[[f32; 4]], ids: &[usize], axis: usize) -> (f32, usize, Vec<usize>) {
        let (lo, hi) = (axis, axis + 2);
        let mut o = ids.to_vec();
        o.sort_by(|&a, &b| boxes[a][lo].total_cmp(&boxes[b][lo]));
        let mut end = boxes[o[0]][hi];
        let (mut best_gap, mut best_k) = (0.0, 0);
        for k in 1..o.len() {
            let gap = boxes[o[k]][lo] - end;
            if gap > best_gap {
                (best_gap, best_k) = (gap, k);
            }
            end = end.max(boxes[o[k]][hi]);
        }
        (best_gap, best_k, o)
    }
    fn rec(boxes: &[[f32; 4]], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let (gy, ky, oy) = best_cut(boxes, &ids, 1);
        let (gx, kx, ox) = best_cut(boxes, &ids, 0);
        if ky == 0 && kx == 0 {
            return top_down(boxes, &ids);
        }
        let (k, o) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let mut out = rec(boxes, o[..k].to_vec());
        out.extend(rec(boxes, o[k..].to_vec()));
        out
    }
    rec(boxes, (0..boxes.len()).collect())
}

/// Lines in reading order into paragraphs: a new one where the gap to the line before is over `gap` median line
/// heights, or the two do not overlap across.
pub(crate) fn paragraphs(lines: &[&Line], gap: f32) -> Vec<Vec<usize>> {
    let med = median(lines.iter().map(|l| l.bounds[3] - l.bounds[1]).collect());
    let mut out: Vec<Vec<usize>> = vec![];
    for (i, l) in lines.iter().enumerate() {
        let new = match i.checked_sub(1).map(|p| lines[p].bounds) {
            None => true,
            Some(p) => {
                let b = l.bounds;
                b[1] - p[3] > gap * med || b[2].min(p[2]) < b[0].max(p[0])
            }
        };
        if new {
            out.push(vec![]);
        }
        out.last_mut().unwrap().push(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, bounds: [f32; 4]) -> Line {
        Line {
            bounds,
            toks: vec![],
            conf: 1.0,
            chars: true,
            text: text.into(),
        }
    }

    #[test]
    fn xycut_reads_columns_before_rows_below() {
        // Two columns, then a full-width line under both.
        let boxes = [
            [0.0, 0.0, 40.0, 10.0],
            [60.0, 0.0, 100.0, 10.0],
            [0.0, 12.0, 40.0, 22.0],
            [60.0, 12.0, 100.0, 22.0],
            [0.0, 40.0, 100.0, 50.0],
        ];
        assert_eq!(xycut(&boxes), vec![0, 2, 1, 3, 4]);
    }

    #[test]
    fn leaders_go_and_markers_join_their_items() {
        let lines = vec![
            line("........", [0.0, 0.0, 100.0, 10.0]),
            line("1.", [0.0, 20.0, 10.0, 30.0]),
            line("First item", [15.0, 20.0, 100.0, 30.0]),
            line("-", [0.0, 40.0, 5.0, 50.0]),
        ];
        let out = tidy(lines, false);
        let texts: Vec<&str> = out.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, vec!["1. First item", "-"]);
        assert!(enumerator("(iv)") && enumerator("12)") && !enumerator("1234."));
    }

    #[test]
    fn paragraphs_break_at_wide_gaps() {
        let a = line("a", [0.0, 0.0, 100.0, 10.0]);
        let b = line("b", [0.0, 12.0, 100.0, 22.0]);
        let c = line("c", [0.0, 40.0, 100.0, 50.0]);
        assert_eq!(paragraphs(&[&a, &b, &c], 0.8), vec![vec![0, 1], vec![2]]);
    }
}

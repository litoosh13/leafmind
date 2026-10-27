//! The printed label of a form field, found from the page's words and their boxes: pure geometry, no model, any
//! language. Boxes are `[left, top, right, bottom]` with y growing downward, in any one unit (PDF points, pixels);
//! the words can come from a PDF's text or from OCR.

/// A word on the page and its box.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub bounds: [f32; 4],
}

/// What kind of field the label is for: a checkbox or radio button usually has its label to the right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelFor {
    Text,
    Choice,
}

/// The label of each field (`None` when nothing near it reads like one), in the order of `fields`.
///
/// Where it looks, in this order: for a text field a short caption in the top half of its box, a caption just
/// below it when that one is in brackets or smaller type ("(Street, house number)" under a fill-in line), the words
/// left of it on the same row (only the line level with its middle; dot leaders are stepped over), the nearest line above it (up to 8 line
/// heights: column headers; two lines when both stand within its column, or when a caption wraps) and a line just below it; for a checkbox or radio button the words right of it, then left, then above. A label never reaches
/// past another field or a box glyph ("□"), and going left it stops at the colon that ends another field's label.
/// A text field in a table row without a label of its own takes the label of the field above it in the same column;
/// comb fields (one box per character) share the label of their first box.
/// Numbering ("1.2", "a)"), a trailing ":" or "*" and fill-in lines go.
pub fn field_labels(fields: &[([f32; 4], LabelFor)], words: &[Word]) -> Vec<Option<String>> {
    let (glyphs, words): (Vec<&Word>, Vec<&Word>) =
        words.iter().partition(|w| w.text.chars().all(is_box_glyph));
    let mut heights: Vec<f32> = words
        .iter()
        .map(|w| w.bounds[3] - w.bounds[1])
        .filter(|h| *h > 0.0)
        .collect();
    heights.sort_by(f32::total_cmp);
    let h = heights.get(heights.len() / 2).copied().unwrap_or(10.0);
    let blockers_of = |i: usize| -> Vec<[f32; 4]> {
        fields
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, f)| f.0)
            .chain(glyphs.iter().map(|g| g.bounds))
            .collect()
    };
    let mut labels: Vec<Option<String>> = fields
        .iter()
        .enumerate()
        .map(|(i, (b, kind))| {
            // Everything a label cannot reach past: the other fields and box glyphs.
            let blockers = blockers_of(i);
            let at = |p| look(p, b, &words, &blockers, h);
            match kind {
                LabelFor::Text => at(Place::Inside)
                    .or_else(|| at(Place::Caption))
                    .or_else(|| at(Place::Left))
                    .or_else(|| at(Place::Above))
                    .or_else(|| at(Place::Below)),
                LabelFor::Choice => at(Place::Right)
                    .or_else(|| at(Place::Left))
                    .or_else(|| at(Place::Above)),
            }
        })
        .collect();
    // Comb fields (IBAN, BIC: one small box per character): a run of 3 or more narrow fields side by side on a row
    // all have the label of the first (boxes may stand up to 1.5 box widths apart: IBAN in groups of four).
    let mut by_left: Vec<usize> = (0..fields.len())
        .filter(|&i| fields[i].1 == LabelFor::Text)
        .collect();
    by_left.sort_by(|&x, &y| fields[x].0[0].total_cmp(&fields[y].0[0]));
    let narrow = |i: usize| fields[i].0[2] - fields[i].0[0] <= 2.5 * h;
    let mut done = vec![false; fields.len()];
    for &first in &by_left {
        if done[first] || !narrow(first) {
            continue;
        }
        let mut run = vec![first];
        loop {
            let last = fields[*run.last().unwrap()].0;
            let next = by_left.iter().copied().find(|&j| {
                let f = fields[j].0;
                !run.contains(&j)
                    && narrow(j)
                    && same_row(&f, &last)
                    && f[0] >= last[2] - 0.3 * h
                    && f[0] - last[2] <= 1.5 * (last[2] - last[0])
            });
            match next {
                Some(j) => run.push(j),
                None => break,
            }
        }
        if run.len() >= 3 {
            for &j in &run[1..] {
                labels[j] = labels[first].clone();
                done[j] = true;
            }
        }
        done[first] = true;
    }
    // Table rows: a text field without a label of its own, right under a field of the same column, has that
    // field's label (the column header), top to bottom.
    let mut order: Vec<usize> = (0..fields.len()).collect();
    order.sort_by(|&x, &y| fields[x].0[1].total_cmp(&fields[y].0[1]));
    for &i in &order {
        let (b, kind) = fields[i];
        if labels[i].is_some() || kind != LabelFor::Text {
            continue;
        }
        let above = order.iter().copied().filter(|&j| {
            let a = fields[j].0;
            let overlap = a[2].min(b[2]) - a[0].max(b[0]);
            j != i
                && b[1] - a[3] >= -0.3 * h
                && b[1] - a[3] <= 2.0 * h
                && overlap >= 0.7 * (b[2] - b[0]).min(a[2] - a[0])
        });
        if let Some(j) =
            above.min_by(|&x, &y| (b[1] - fields[x].0[3]).total_cmp(&(b[1] - fields[y].0[3])))
        {
            labels[i] = labels[j].clone();
        }
    }
    labels
}

#[derive(Clone, Copy, PartialEq)]
enum Place {
    Inside,
    /// Below, but only a caption in brackets or smaller type.
    Caption,
    Left,
    Right,
    Above,
    Below,
}

/// The texts around each field, each with where it stands: "inside" (a caption in the top half of its box), "left",
/// "right", "above" and "below" — every place that has text, not one chosen label. For matching a request against
/// the fields (which field does "my phone number is …" mean?) when no single label can be told reliably.
pub fn field_context(
    fields: &[([f32; 4], LabelFor)],
    words: &[Word],
) -> Vec<Vec<(&'static str, String)>> {
    let (glyphs, words): (Vec<&Word>, Vec<&Word>) =
        words.iter().partition(|w| w.text.chars().all(is_box_glyph));
    let mut heights: Vec<f32> = words
        .iter()
        .map(|w| w.bounds[3] - w.bounds[1])
        .filter(|h| *h > 0.0)
        .collect();
    heights.sort_by(f32::total_cmp);
    let h = heights.get(heights.len() / 2).copied().unwrap_or(10.0);
    fields
        .iter()
        .enumerate()
        .map(|(i, (b, _))| {
            let blockers: Vec<[f32; 4]> = fields
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, f)| f.0)
                .chain(glyphs.iter().map(|g| g.bounds))
                .collect();
            [
                ("inside", Place::Inside),
                ("left", Place::Left),
                ("right", Place::Right),
                ("above", Place::Above),
                ("below", Place::Below),
            ]
            .into_iter()
            .filter_map(|(name, p)| look(p, b, &words, &blockers, h).map(|t| (name, t)))
            .collect()
        })
        .collect()
}

/// Dots or underscores only: a leader between a label and its field (". . . . ."), not text.
fn is_leader(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| matches!(c, '.' | '…' | '·' | '_'))
}

/// Checkbox and radio glyphs that PDFs and OCR give as text (the last two: Wingdings boxes in the private-use area).
fn is_box_glyph(c: char) -> bool {
    matches!(
        c,
        '□' | '☐'
            | '☑'
            | '☒'
            | '■'
            | '▢'
            | '◻'
            | '❏'
            | '○'
            | '●'
            | '◯'
            | '⊠'
            | '\u{F06F}'
            | '\u{F0A8}'
    )
}

/// Vertical overlap of two boxes, as a share of the lower one.
fn same_row(a: &[f32; 4], b: &[f32; 4]) -> bool {
    let overlap = a[3].min(b[3]) - a[1].max(b[1]);
    overlap >= 0.5 * (a[3] - a[1]).min(b[3] - b[1]).max(0.1)
}

/// Whether a blocker lies in the rectangle between `x0..x1` and `y0..y1` (a label cannot reach across it).
fn crossed(blockers: &[[f32; 4]], x0: f32, x1: f32, y0: f32, y1: f32) -> bool {
    blockers
        .iter()
        .any(|k| k[0] < x1 && k[2] > x0 && k[1] < y1 && k[3] > y0)
}

/// The words at `place` of field box `b`, as one cleaned label.
fn look(
    place: Place,
    b: &[f32; 4],
    words: &[&Word],
    blockers: &[[f32; 4]],
    h: f32,
) -> Option<String> {
    let gap = 1.5 * h; // words further apart than this are not one label
    let mid = (b[1] + b[3]) / 2.0;
    let picked: Vec<&Word> = match place {
        Place::Inside => {
            let top_half = b[1] + 0.5 * (b[3] - b[1]);
            let inside: Vec<&Word> = words
                .iter()
                .copied()
                .filter(|w| {
                    let c = [
                        (w.bounds[0] + w.bounds[2]) / 2.0,
                        (w.bounds[1] + w.bounds[3]) / 2.0,
                    ];
                    c[0] > b[0] && c[0] < b[2] && c[1] > b[1] && c[1] < top_half
                })
                .collect();
            if inside.len() > 8 {
                return None;
            }
            inside
        }
        Place::Left | Place::Right => {
            let mut row: Vec<&Word> = words
                .iter()
                .copied()
                .filter(|w| same_row(&w.bounds, b))
                .collect();
            // A field as tall as two text lines: only the line level with its middle, on the side searched.
            let centre = |w: &Word| (w.bounds[1] + w.bounds[3]) / 2.0;
            let left = place == Place::Left;
            row.retain(|w| {
                if left {
                    w.bounds[2] <= b[0] + 0.5 * h
                } else {
                    w.bounds[0] >= b[2] - 0.5 * h
                }
            });
            if let Some(level) = row
                .iter()
                .map(|w| centre(w))
                .min_by(|x, y| (x - mid).abs().total_cmp(&(y - mid).abs()))
            {
                row.retain(|w| (centre(w) - level).abs() < 0.5 * h);
            }
            row.sort_by(|x, y| x.bounds[0].total_cmp(&y.bounds[0]));
            let mut run = Vec::new();
            if place == Place::Left {
                let mut edge = b[0];
                for w in row.iter().rev().filter(|w| w.bounds[2] <= b[0] + 0.5 * h) {
                    // The first word may sit up to 15 line heights away (labels in a column); then words must touch.
                    let limit = if run.is_empty() { 15.0 * h } else { gap };
                    let ends_label = !run.is_empty() && w.text.ends_with([':', '：']);
                    if edge - w.bounds[2] > limit
                        || ends_label
                        || crossed(blockers, w.bounds[2], edge, mid - 0.1, mid + 0.1)
                    {
                        break;
                    }
                    // Dot leaders (". . . . 1a $") lead to the label: step over them.
                    if is_leader(&w.text) {
                        edge = w.bounds[0];
                        continue;
                    }
                    run.push(*w);
                    edge = w.bounds[0];
                }
                run.reverse();
            } else {
                let mut edge = b[2];
                for w in row.iter().filter(|w| w.bounds[0] >= b[2] - 0.5 * h) {
                    let limit = if run.is_empty() { 3.0 * h } else { gap };
                    if w.bounds[0] - edge > limit
                        || crossed(blockers, edge, w.bounds[0], mid - 0.1, mid + 0.1)
                    {
                        break;
                    }
                    run.push(*w);
                    edge = w.bounds[2];
                }
            }
            run
        }
        Place::Above | Place::Below | Place::Caption => {
            let above = place == Place::Above;
            // Above, a column header may stand several rows up (nothing may lie in between).
            let reach = if above { 8.0 * h } else { 1.5 * h };
            let dist = |w: &Word| {
                if above {
                    b[1] - w.bounds[3]
                } else {
                    w.bounds[1] - b[3]
                }
            };
            let mut cands: Vec<&Word> = words
                .iter()
                .copied()
                .filter(|w| {
                    let d = dist(w);
                    d >= -0.3 * h && d <= reach && w.bounds[2] > b[0] - h && w.bounds[0] < b[2]
                })
                .filter(|w| {
                    let (y0, y1) = if above {
                        (w.bounds[3], b[1])
                    } else {
                        (b[3], w.bounds[1])
                    };
                    !crossed(
                        blockers,
                        w.bounds[0].max(b[0]),
                        w.bounds[2].min(b[2]),
                        y0 + 0.1,
                        y1 - 0.1,
                    )
                })
                .collect();
            // Only the nearest line; above, also the line over it when both stand within this field's column (a
            // header on two lines: "Relationship / to the applicant"), not a paragraph running across the page.
            if let Some(first) = cands.iter().map(|w| dist(w)).min_by(f32::total_cmp) {
                let near: Vec<&Word> = cands
                    .iter()
                    .copied()
                    .filter(|w| dist(w) - first < 0.5 * h)
                    .collect();
                let next: Vec<&Word> = cands
                    .iter()
                    .copied()
                    .filter(|w| dist(w) - first >= 0.5 * h && dist(w) - first <= 1.8 * h)
                    .collect();
                let local = |line: &[&Word]| {
                    line.first().is_some_and(|l| {
                        words
                            .iter()
                            .filter(|w| same_row(&w.bounds, &l.bounds))
                            .all(|w| w.bounds[0] >= b[0] - h && w.bounds[2] <= b[2] + h)
                    })
                };
                // A caption wrapped onto two lines: the line over the nearest one starts at the same left edge.
                let left_edge =
                    |line: &[&Word]| line.iter().map(|w| w.bounds[0]).fold(f32::MAX, f32::min);
                let wrapped =
                    |near: &[&Word], next: &[&Word]| (left_edge(near) - left_edge(next)).abs() <= h;
                cands = if above
                    && !next.is_empty()
                    && ((near.len() <= 4 && local(&near) && local(&next)) || wrapped(&near, &next))
                {
                    near.into_iter().chain(next).collect()
                } else {
                    near
                };
            }
            if place == Place::Caption {
                let small = cands.iter().all(|w| w.bounds[3] - w.bounds[1] <= 0.85 * h);
                let bracketed = cands
                    .iter()
                    .min_by(|x, y| x.bounds[0].total_cmp(&y.bounds[0]))
                    .is_some_and(|w| w.text.starts_with('('));
                if !(small || bracketed) {
                    return None;
                }
            }
            cands
        }
    };
    // Reading order: lines top to bottom (words whose tops differ by less than half a line are one line, as
    // italics and punctuation sit a little higher or lower), each line left to right.
    let mut picked = picked;
    picked.sort_by(|x, y| x.bounds[1].total_cmp(&y.bounds[1]));
    let mut line = 0;
    let mut lines: Vec<(usize, &Word)> = Vec::with_capacity(picked.len());
    for (k, w) in picked.iter().enumerate() {
        if k > 0 && w.bounds[1] - picked[k - 1].bounds[1] > 0.5 * h {
            line += 1;
        }
        lines.push((line, w));
    }
    lines.sort_by(|(la, a), (lb, b)| la.cmp(lb).then(a.bounds[0].total_cmp(&b.bounds[0])));
    let picked: Vec<&Word> = lines.into_iter().map(|(_, w)| w).collect();
    let text = picked
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    clean(&text)
}

/// A label without numbering, fill-in lines and trailing ":" / "*"; `None` when no letters are left.
fn clean(text: &str) -> Option<String> {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| !w.chars().all(|c| matches!(c, '_' | '.' | '…' | '-')))
        .collect();
    let mut words = words.as_slice();
    // Leading numbering: "1.", "2.3", "(4)", "a)".
    let numbering = |f: &str| {
        let core = f.trim_start_matches('(').trim_end_matches(['.', ')']);
        let digits = !core.is_empty() && core.chars().all(|c| c.is_ascii_digit() || c == '.');
        (digits && f.chars().count() <= 5) || (f.chars().count() == 2 && f.ends_with(')'))
    };
    if words.first().is_some_and(|f| numbering(f)) {
        words = &words[1..];
    }
    // Sideways text comes out as single letters ("a t S"): no label.
    if words.len() > 1 && words.iter().all(|w| w.chars().count() == 1) {
        return None;
    }
    let label = words.join(" ");
    let label = label.trim_end_matches([':', '*', ' ', '：']).trim();
    let label = label
        .strip_prefix('(')
        .and_then(|l| l.strip_suffix(')'))
        .unwrap_or(label)
        .trim();
    // At least two letters: a lone letter is sideways text or a mark.
    (label.chars().filter(|c| c.is_alphabetic()).count() >= 2).then(|| label.to_string())
}

/// A form field's tooltip (PDF /TU) as its label, when it reads like one: authors often put the field's label there.
/// `None` for generic names ("Text3", "Check Box12", "undefined") and for long help texts (over 80 characters);
/// underscores become spaces and a table-row suffix ("_Row_2") goes. Prefer it over [`field_labels`] when present.
pub fn tooltip_label(tooltip: &str) -> Option<String> {
    let t = tooltip.replace('_', " ");
    let mut t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some((head, n)) = t.rsplit_once(" Row ")
        && n.chars().all(|c| c.is_ascii_digit())
    {
        t = head.to_string();
    }
    let lower = t.to_lowercase();
    let generic = [
        "text",
        "textfeld",
        "check box",
        "checkbox",
        "kontrollk",
        "topmostsubform",
        "formular",
        "field",
        "feld",
        "untitled",
        "undefined",
    ]
    .iter()
    .any(|g| {
        lower.starts_with(g)
            && lower[g.len()..]
                .chars()
                .all(|c| c.is_ascii_digit() || c == ' ' || c == '.' || c == '[' || c == ']')
    });
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    (!generic && letters >= 3 && t.chars().count() <= 80).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, l: f32, t: f32) -> Word {
        Word {
            text: text.into(),
            bounds: [l, t, l + 6.0 * text.chars().count() as f32, t + 10.0],
        }
    }

    #[test]
    fn tooltips_that_read_like_labels() {
        assert_eq!(
            tooltip_label("Telefax (Mit Vorwahl)*"),
            Some("Telefax (Mit Vorwahl)*".into())
        );
        assert_eq!(
            tooltip_label("Anzahl_Hausperling"),
            Some("Anzahl Hausperling".into())
        );
        assert_eq!(
            tooltip_label("Lage am Gebäude_Row_2"),
            Some("Lage am Gebäude".into())
        );
        for generic in ["Text3", "Check Box12", "undefined", "Textfeld 7", "[1]", ""] {
            assert_eq!(tooltip_label(generic), None, "{generic}");
        }
        assert_eq!(tooltip_label(&"Bitte geben Sie hier ".repeat(5)), None);
    }

    #[test]
    fn labels_left_right_above_inside() {
        let words = vec![
            w("1.2", 50.0, 100.0),
            w("Date", 70.0, 100.0),
            w("of", 100.0, 100.0),
            w("birth:", 115.0, 100.0),
            w("Newsletter", 215.0, 150.0),
            w("Street", 300.0, 188.0),
            w("Town", 52.0, 252.0),
        ];
        let fields = [
            ([160.0, 98.0, 300.0, 112.0], LabelFor::Text), // right of "Date of birth:"
            ([200.0, 150.0, 210.0, 160.0], LabelFor::Choice), // checkbox before "Newsletter"
            ([300.0, 200.0, 450.0, 214.0], LabelFor::Text), // under the caption "Street"
            ([50.0, 250.0, 200.0, 275.0], LabelFor::Text), // box with the caption "Town" inside
            ([600.0, 600.0, 700.0, 614.0], LabelFor::Text), // nothing near
        ];
        assert_eq!(
            field_labels(&fields, &words),
            [
                Some("Date of birth".into()),
                Some("Newsletter".into()),
                Some("Street".into()),
                Some("Town".into()),
                None
            ]
        );
    }
}

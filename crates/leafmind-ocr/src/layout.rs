//! Turns Tesseract's word list (TSV) into page text: words → lines → paragraphs. leafmind groups the
//! paragraphs itself because paragraph size matters for question answering, and Tesseract's own paragraphs
//! split more finely than a PDF's.

/// A new paragraph starts at a new text block, or when the space above a line is larger than this many median
/// line heights. (On the benchmark scans Tesseract's blocks already give the PDF's paragraphs; any value from
/// 0.5 to "never" changed nothing.)
const PARAGRAPH_GAP: f32 = 0.9;

#[derive(Debug)]
struct Line {
    block: u32,
    top: f32,
    bottom: f32,
    left: f32,
    right: f32,
    words: Vec<String>,
}

/// One word row of the TSV: (block, paragraph, line, left, top, width, height, confidence, text).
fn words(tsv: &str) -> impl Iterator<Item = (u32, u32, u32, f32, f32, f32, f32, f32, &str)> {
    tsv.lines().filter_map(|row| {
        let c: Vec<&str> = row.split('\t').collect();
        if c.len() < 12 || c[0] != "5" {
            return None;
        }
        let n = |i: usize| c[i].trim().parse::<f32>().ok();
        let text = c[11].trim();
        if text.is_empty() {
            return None;
        }
        Some((
            c[2].parse().ok()?,
            c[3].parse().ok()?,
            c[4].parse().ok()?,
            n(6)?,
            n(7)?,
            n(8)?,
            n(9)?,
            n(10)?,
            text,
        ))
    })
}

/// Page text with paragraphs separated by a blank line, and the median word confidence (0–1).
pub(crate) fn page_text(tsv: &str) -> (String, f32) {
    let mut lines: Vec<Line> = Vec::new();
    let mut key = None;
    let mut confidences = Vec::new();
    for (block, par, line, left, top, width, height, conf, text) in words(tsv) {
        confidences.push(conf);
        if key != Some((block, par, line)) {
            key = Some((block, par, line));
            lines.push(Line {
                block,
                top,
                bottom: top + height,
                left,
                right: left + width,
                words: Vec::new(),
            });
        }
        let current = lines.last_mut().unwrap();
        current.top = current.top.min(top);
        current.bottom = current.bottom.max(top + height);
        current.left = current.left.min(left);
        current.right = current.right.max(left + width);
        current.words.push(text.to_string());
    }
    let mut heights: Vec<f32> = lines.iter().map(|l| l.bottom - l.top).collect();
    heights.sort_by(f32::total_cmp);
    let median = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
    confidences.sort_by(f32::total_cmp);
    let confidence = confidences
        .get(confidences.len() / 2)
        .map_or(0.0, |c| c / 100.0);
    if let Some(text) = form_rows(&lines, median) {
        return (text, confidence);
    }
    let mut text = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            let prev = &lines[i - 1];
            let gap = line.top - prev.bottom;
            text.push_str(
                if line.block != prev.block || gap > PARAGRAPH_GAP * median {
                    "\n\n"
                } else {
                    " "
                },
            );
        }
        text.push_str(&line.words.join(" "));
    }
    (text, confidence)
}

/// Whether text is mostly right-to-left (Arabic script).
fn is_rtl(text: &str) -> bool {
    let rtl = text.chars().filter(|c| matches!(c, '\u{0600}'..='\u{06FF}' | '\u{0750}'..='\u{077F}' | '\u{FB50}'..='\u{FEFF}')).count();
    rtl * 2 > text.chars().filter(|c| c.is_alphabetic()).count()
}

/// A form read row by row. Tesseract puts a form's labels and its filled-in values in different text blocks
/// when they stand in columns, so block by block the labels come out together and the values elsewhere. If the
/// page looks like such a grid — at least 3 rows, and at least a quarter of all rows, where a short label
/// (≤ 40 characters, a real word) in one block has text of another block at least two line heights further along, and these
/// are at least 60 % of all rows with text of two blocks side by side (two columns of prose are not) — the
/// lines are regrouped into rows (lines whose heights overlap by half), read across in their script's
/// direction, as "label: value". Rows further apart than two line heights start a new paragraph; very tall
/// lines (sideways text) follow as paragraphs of their own.
fn form_rows(lines: &[Line], median: f32) -> Option<String> {
    if median <= 0.0 {
        return None;
    }
    let (mut flat, tall): (Vec<&Line>, Vec<&Line>) =
        lines.iter().partition(|l| l.bottom - l.top <= 3.0 * median);
    flat.sort_by(|a, b| a.top.total_cmp(&b.top));
    let mut rows: Vec<Vec<&Line>> = Vec::new();
    for line in flat {
        match rows.last_mut() {
            Some(row)
                if row[0].bottom.min(line.bottom) - row[0].top.max(line.top)
                    >= 0.5 * (row[0].bottom - row[0].top).min(line.bottom - line.top) =>
            {
                row.push(line)
            }
            _ => rows.push(vec![line]),
        }
    }
    // Each row in reading order, with the gap before each line (0 for the first).
    let read: Vec<(Vec<(&Line, f32)>, bool)> = rows
        .iter()
        .map(|row| {
            let text: String = row.iter().map(|l| l.words.join(" ")).collect();
            let rtl = is_rtl(&text);
            let mut row = row.clone();
            row.sort_by(|a, b| {
                if rtl {
                    b.right.total_cmp(&a.right)
                } else {
                    a.left.total_cmp(&b.left)
                }
            });
            let gaps = (0..row.len()).map(|i| match i {
                0 => 0.0,
                _ if rtl => row[i - 1].left - row[i].right,
                _ => row[i].left - row[i - 1].right,
            });
            (
                row.iter().copied().zip(gaps.collect::<Vec<_>>()).collect(),
                rtl,
            )
        })
        .collect();
    let wide = 2.0 * median;
    // Rows with text of another block well along the line ("split"), and those among them whose first part is
    // a short label. Two columns of prose split every row too, but their left parts are long column lines.
    let chars = |part: &[(&Line, f32)]| {
        part.iter()
            .map(|(l, _)| l.words.join(" ").chars().count())
            .sum::<usize>()
    };
    let (mut split, mut labelled) = (0, 0);
    for (row, _) in &read {
        let cuts: Vec<usize> = (1..row.len())
            .filter(|&i| row[i].1 >= wide && row[i].0.block != row[i - 1].0.block)
            .collect();
        if let Some(&first) = cuts.first() {
            split += 1;
            let label: String = row[..first]
                .iter()
                .map(|(l, _)| l.words.join(" "))
                .collect::<Vec<_>>()
                .join(" ");
            // Scanner noise (bits of sideways margin text) reads as "£", "Ss", "8": a label has 3 letters, or 2
            // and a closing ":" ("Nr.:").
            let letters = label.chars().filter(|c| c.is_alphabetic()).count();
            let word = letters >= 3 || (letters >= 2 && label.trim_end().ends_with([':', '：']));
            labelled += usize::from(word && chars(&row[..first]) <= 40);
        }
    }
    if labelled < 3 || labelled * 4 < rows.len() || labelled * 5 < split * 3 {
        return None;
    }
    let mut text = String::new();
    let mut prev_bottom: Option<f32> = None;
    for (row, _) in &read {
        let top = row.iter().map(|(l, _)| l.top).fold(f32::MAX, f32::min);
        if let Some(bottom) = prev_bottom {
            text.push_str(if top - bottom > wide { "\n\n" } else { "\n" });
        }
        let mut label_done = false;
        for (i, (line, gap)) in row.iter().enumerate() {
            if i > 0 {
                let label = !label_done && *gap >= wide && !text.ends_with([':', '：']);
                text.push_str(if label { ": " } else { " " });
                label_done |= *gap >= wide;
            }
            text.push_str(&line.words.join(" "));
        }
        prev_bottom = Some(row.iter().map(|(l, _)| l.bottom).fold(f32::MIN, f32::max));
    }
    for line in tall {
        text.push_str("\n\n");
        text.push_str(&line.words.join(" "));
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        block: u32,
        par: u32,
        line: u32,
        top: u32,
        height: u32,
        conf: u32,
        text: &str,
    ) -> String {
        format!("5\t1\t{block}\t{par}\t{line}\t1\t10\t{top}\t50\t{height}\t{conf}\t{text}\n")
    }

    #[test]
    fn lines_and_paragraphs() {
        let header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n";
        let tsv = [
            header.to_string(),
            "1\t1\t0\t0\t0\t0\t0\t0\t800\t1000\t-1\t\n".into(), // page row, ignored
            row(1, 1, 1, 100, 20, 95, "Garden"),
            row(1, 1, 1, 100, 20, 93, "rules"),
            row(2, 1, 1, 160, 20, 90, "The"),
            row(2, 1, 1, 160, 20, 90, "rent"),
            row(2, 1, 2, 184, 20, 80, "is"),
            row(2, 1, 2, 184, 20, 70, "120 Euro."),
            row(2, 2, 1, 240, 20, 96, "Water"),
            row(2, 2, 1, 240, 20, 96, "is included."),
            row(2, 2, 1, 240, 20, 50, " "), // empty word, ignored
        ]
        .concat();
        let (text, confidence) = page_text(&tsv);
        assert_eq!(
            text,
            "Garden rules\n\nThe rent is 120 Euro.\n\nWater is included."
        );
        assert_eq!(confidence, 0.93);
    }

    #[test]
    fn nothing_recognised() {
        assert_eq!(page_text("level\tpage_num\n"), (String::new(), 0.0));
    }

    /// One word at a position: (block, line, left, top, text); words of a line are given in reading order.
    fn tsv(words: &[(u32, u32, u32, u32, &str)]) -> String {
        words
            .iter()
            .map(|&(b, l, left, top, t)| {
                let width = 9 * t.chars().count() as u32;
                format!("5\t1\t{b}\t1\t{l}\t1\t{left}\t{top}\t{width}\t20\t90\t{t}\n")
            })
            .collect()
    }

    #[test]
    fn a_form_grid_is_read_row_by_row() {
        // Labels in block 1, the filled-in values in block 2, as Tesseract splits a form in columns.
        let mut words = Vec::new();
        let rows = [
            ("Plot:", "B-14"),
            ("Date:", "14.05.2027"),
            ("Member:", "4711"),
            ("Rent:", "120 Euro"),
        ];
        for (k, (label, value)) in rows.iter().enumerate() {
            let top = 100 + 30 * k as u32;
            words.push((1, k as u32, 100, top, *label));
            words.push((2, k as u32, 400, top + 2, *value));
        }
        let (text, _) = page_text(&tsv(&words));
        assert_eq!(
            text,
            "Plot: B-14\nDate: 14.05.2027\nMember: 4711\nRent: 120 Euro"
        );
    }

    #[test]
    fn prose_columns_and_margin_noise_stay_as_they_are() {
        // Two columns of prose (long lines), plus bits of sideways margin text read as noise ("£", "8").
        // Tesseract lists one block after the other.
        let mut words = Vec::new();
        for (block, left, text) in [
            (1, 100, "the left column runs on with a long line"),
            (2, 600, "and the right one does the same here too"),
            (3, 20, "£"),
        ] {
            for k in 0..6u32 {
                words.push((
                    block,
                    k,
                    left,
                    100 + 30 * k,
                    if block == 3 && k % 2 == 1 { "8" } else { text },
                ));
            }
        }
        let (text, _) = page_text(&tsv(&words));
        assert!(
            text.contains("the left column runs on with a long line the left column"),
            "{text}"
        );
    }
}

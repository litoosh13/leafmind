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
    for (block, par, line, _left, top, _width, height, conf, text) in words(tsv) {
        confidences.push(conf);
        if key != Some((block, par, line)) {
            key = Some((block, par, line));
            lines.push(Line {
                block,
                top,
                bottom: top + height,
                words: Vec::new(),
            });
        }
        let current = lines.last_mut().unwrap();
        current.top = current.top.min(top);
        current.bottom = current.bottom.max(top + height);
        current.words.push(text.to_string());
    }
    let mut heights: Vec<f32> = lines.iter().map(|l| l.bottom - l.top).collect();
    heights.sort_by(f32::total_cmp);
    let median = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
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
    confidences.sort_by(f32::total_cmp);
    let confidence = confidences
        .get(confidences.len() / 2)
        .map_or(0.0, |c| c / 100.0);
    (text, confidence)
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
}

//! Cutting page text into paragraphs and sentences.

use crate::{Error, Passage};
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Words that mark an amendment (a later change to a clause), in English, German, Persian and Arabic.
static AMENDMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)amend|nachtrag|änderung|ab dem|with effect|effective|الحاقیه|اصلاحیه|ملحق|اعتباراً|تعديل").unwrap()
});

/// A sentence piece ending like this is an abbreviation, an initial or an ordinal ("Nr.", "Silas B.", "e.g.",
/// "z. B.", "1."), not a sentence end. Words that often end a sentence ("etc.", "usw.", "Inc.") are not listed.
static ABBREVIATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?:\b(?:No|Nr|Abs|Art|bzw|ca|Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr|vs|Vol|Fig|Figs|Eq|Ref|Refs|al|approx",
        r"|Abb|Kap|vgl|evtl|ggf|inkl|zzgl|Tel|Str|Mio|Mrd|Tsd|Jh)\.",
        // A single letter: initials, and the last letter of "e.g.", "i.e.", "z. B.", "d. h.", "u. a."
        r"|\b\p{L}\.",
        r"|\b\d+\.)$"
    ))
    .unwrap()
});

/// Symbols that start a list item when they stand alone between spaces: common bullets, and the letters
/// that Word's Symbol/Wingdings bullets become when a PDF maps them to text ("Ø", "ç", "ü", "v").
const BULLETS: [char; 17] = [
    '•', '●', '▪', '◦', '‣', '∙', '■', '□', '✓', '✔', '➢', '➤', '►', 'Ø', 'ç', 'ü', 'v',
];

/// Empty fill-in lines of a form ("________", "___.___.______", ". . . ." leaders are not touched).
static PLACEHOLDER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[_.]*_{2,}[_.]*|\.{4,}").unwrap());

/// A numbered form or contract label ("1.2 Date of birth", "3. Rent"); "01.05.2027" or "0170 123" are not.
static NUMBERED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d+\.(\d+\.?)*\s").unwrap());

/// Form text made readable: table rows become sentences (see [`table_rows`]) and empty fill-in lines go.
fn form_text(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut table: Vec<&str> = Vec::new();
    for line in text.lines().map(str::trim).chain([""]) {
        if line.starts_with('|') {
            table.push(line);
            continue;
        }
        out.extend(table_rows(&table));
        table.clear();
        let line = PLACEHOLDER.replace_all(line, " ");
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() {
            out.push(line);
        }
    }
    out.join("\n")
}

/// A markdown table as one line per row. Two columns are a form's label and value ("Name: Maria Example");
/// with three or more, the first row names the columns ("Plan: Premium, Uptime: 99.9%").
fn table_rows(lines: &[&str]) -> Vec<String> {
    let rows: Vec<Vec<&str>> = lines
        .iter()
        .map(|l| {
            l.split('|')
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|cells| {
            !cells
                .iter()
                .all(|c| c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
        })
        .collect();
    match rows.split_first() {
        Some((header, body)) if header.len() >= 3 && !body.is_empty() => body
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, cell)| match header.get(i) {
                        Some(name) => format!("{name}: {cell}"),
                        None => cell.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .collect(),
        _ => rows
            .iter()
            .map(|row| match row.split_first() {
                Some((label, rest)) if !rest.is_empty() => format!("{label}: {}", rest.join(", ")),
                _ => row.join(" "),
            })
            .collect(),
    }
}

/// Whether a line starts with a list bullet standing alone ("• gloves", "v gloves"; not "vierzig").
fn starts_with_bullet(line: &str) -> bool {
    let mut chars = line.chars();
    chars.next().is_some_and(|c| BULLETS.contains(&c))
        && chars.next().is_none_or(char::is_whitespace)
}

/// A short line without a sentence end or a ":" of its own, like a form label ("1.2 Date of birth") waiting
/// for its value, or a numbered heading.
fn is_bare_label(line: &str) -> bool {
    line.chars().count() <= 60
        && line.chars().any(char::is_alphabetic)
        && !line.contains([':', '：'])
        && !line.ends_with(['.', '!', '?', '؟', ';', ',', '،'])
}

/// Whether `value` (a line or paragraph) is the value of the bare label before it, not a label, heading text
/// or list item of its own.
fn joins_label(label: &str, value: &str) -> bool {
    is_bare_label(label)
        && !label.contains('\n')
        && !value.contains('\n')
        && !NUMBERED.is_match(value)
        && !starts_with_bullet(value)
        && !value.ends_with(['.', '!', '?', '؟'])
        && value.chars().count() <= 120
}

/// Removes the markdown pdf-inspector writes around bold and underlined words.
fn strip_markup(text: &str) -> String {
    text.replace("**", "")
        .replace("<u>", "")
        .replace("</u>", "")
}

/// Splits text at list bullets that stand alone; the bullets themselves are dropped.
fn list_items(text: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut items = Vec::new();
    let mut moved: Vec<(usize, &str)> = Vec::new();
    let mut start = 0;
    for (k, &(i, c)) in chars.iter().enumerate() {
        let alone = (k == 0 || chars[k - 1].1.is_whitespace())
            && chars.get(k + 1).is_none_or(|n| n.1.is_whitespace());
        if alone && BULLETS.contains(&c) {
            // Right-to-left lines often come out of PDFs with a number moved in front of the bullet
            // ("16 Ø متر" for "Ø 16 متر"): a bare number there belongs to the item after the bullet.
            let before = text[start..i].trim_end();
            let number = before.rsplit(char::is_whitespace).next().unwrap_or("");
            let rtl_next = text[i..]
                .chars()
                .find(|c| c.is_alphabetic() && *c != 'Ø' && *c != 'ç')
                .is_some_and(is_rtl_letter);
            if rtl_next
                && !number.is_empty()
                && number.chars().all(|c| c.is_numeric() || "%/.,".contains(c))
            {
                items.push(&text[start..start + before.len() - number.len()]);
                moved.push((items.len(), number));
            } else {
                items.push(&text[start..i]);
            }
            start = i + c.len_utf8();
        }
    }
    items.push(&text[start..]);
    items
        .into_iter()
        .enumerate()
        .map(|(n, item)| match moved.iter().find(|(at, _)| *at == n) {
            Some((_, number)) => format!("{number} {}", item.trim_start()),
            None => item.to_string(),
        })
        .collect()
}

/// Arabic-script (Arabic, Persian, Urdu) or Hebrew letter.
fn is_rtl_letter(c: char) -> bool {
    matches!(c, '\u{0590}'..='\u{08FF}' | '\u{FB1D}'..='\u{FDFF}' | '\u{FE70}'..='\u{FEFF}')
}

/// The text of every page of a PDF as markdown, with 1-based page numbers, read by pdf-inspector (with
/// leafmind's patch that keeps the Persian zero-width non-joiner). Numbers stay exactly as the PDF has them.
/// A form laid out as a grid (labels on the left, values to their right) is read row by row instead, see
/// [`form_rows`].
pub fn pdf_pages(pdf: &[u8]) -> Result<Vec<(u32, String)>, Error> {
    let result = pdf_inspector::extract_pages_markdown_mem(pdf, None)
        .map_err(|e| Error::Pdf(e.to_string()))?;
    // pdf-inspector reads a page with two text columns column by column: right for an article, wrong for a
    // form, where each label's value sits to its right. Pages it found tables on are already read by row.
    let grid: Vec<u32> = result
        .pages_with_columns
        .iter()
        .filter(|p| !result.pages_with_tables.contains(p))
        .copied()
        .collect();
    use pdf_inspector::types::ItemType;
    let items: Vec<(u32, Piece)> = match grid.is_empty() {
        true => Vec::new(),
        false => pdf_inspector::extract_text_with_positions_mem(pdf)
            .unwrap_or_default()
            .into_iter()
            // Sideways margin notes (a form number, "tick where applicable") would tie many rows together.
            .filter(|i| {
                matches!(i.item_type, ItemType::Text | ItemType::FormField)
                    && i.rotation.abs() < 1.0
                    && !i.text.trim().is_empty()
            })
            .map(|i| {
                let field = matches!(i.item_type, ItemType::FormField);
                let piece = Piece {
                    x: i.x,
                    y: i.y,
                    width: i.width,
                    height: i.height,
                    text: i.text,
                    field,
                };
                (i.page, piece)
            })
            .collect(),
    };
    // ponytail: a scanned page (no text layer) comes back empty; say so to the user once OCR exists
    Ok(result
        .pages
        .into_iter()
        .map(|p| {
            let page = p.page + 1;
            let rows = grid
                .contains(&page)
                .then(|| form_rows(items.iter().filter(|i| i.0 == page).map(|i| &i.1).collect()));
            (page, rows.flatten().unwrap_or(p.markdown))
        })
        .collect())
}

/// A piece of text on a page: its box in PDF points (y grows upward, `y` is the bottom) and whether it is the
/// value of a filled form field.
struct Piece {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    text: String,
    field: bool,
}

/// One visual row of a page: its text pieces from left to right.
struct Row<'a> {
    top: f32,
    bottom: f32,
    items: Vec<&'a Piece>,
}

/// A page's text read row by row, if the page looks like a form grid: at least 5 rows, and at least a third of
/// them a short label (≤ 40 characters) with more text well to its right. Each row becomes a line
/// "label: value"; rows further apart than two lines start a new paragraph.
fn form_rows(mut items: Vec<&Piece>) -> Option<String> {
    // The page's usual type size: gaps and headings are measured in it.
    let mut sizes: Vec<f32> = items
        .iter()
        .filter(|i| !i.field)
        .map(|i| i.height)
        .collect();
    sizes.sort_by(f32::total_cmp);
    let line = sizes.get(sizes.len() / 2).copied().unwrap_or(10.0).max(1.0);
    // Rows: items whose heights overlap by at least half of the smaller one, top to bottom.
    items.sort_by(|a, b| (b.y + b.height).total_cmp(&(a.y + a.height)));
    let mut rows: Vec<Row> = Vec::new();
    for item in items {
        let (top, bottom) = (item.y + item.height.max(1.0), item.y);
        match rows.last_mut() {
            Some(row)
                if row.top.min(top) - row.bottom.max(bottom)
                    >= 0.5 * (row.top - row.bottom).min(top - bottom) =>
            {
                row.top = row.top.max(top);
                row.bottom = row.bottom.min(bottom);
                row.items.push(item);
            }
            _ => rows.push(Row {
                top,
                bottom,
                items: vec![item],
            }),
        }
    }
    // A label's value starts at least two type sizes to its right (digits spaced out to fit boxes are closer).
    let gap = 2.0 * line;
    let gap_after =
        |row: &Row, k: usize| row.items[k + 1].x - (row.items[k].x + row.items[k].width);
    for row in &mut rows {
        row.items.sort_by(|a, b| a.x.total_cmp(&b.x));
    }
    let labelled = rows
        .iter()
        .filter(|row| {
            (0..row.items.len().saturating_sub(1)).any(|k| {
                gap_after(row, k) >= gap
                    && row.items[..=k]
                        .iter()
                        .map(|i| i.text.trim().chars().count())
                        .sum::<usize>()
                        <= 40
            })
        })
        .count();
    if rows.len() < 5 || labelled * 3 < rows.len() {
        return None;
    }
    // Type size of a row: its largest text (form fields are boxes, often taller than the text).
    let size = |row: &Row| {
        row.items
            .iter()
            .filter(|i| !i.field)
            .map(|i| i.height)
            .fold(0.0f32, f32::max)
    };
    let mut text = String::new();
    let mut prev: Option<(f32, bool)> = None; // bottom of the last row, and whether it was a heading
    for row in &rows {
        let mut line_text = String::new();
        let mut label_done = false;
        let mut end = f32::MIN; // right edge of the last piece kept
        for item in &row.items {
            let piece = PLACEHOLDER.replace_all(item.text.trim(), " ");
            let piece = piece.split_whitespace().collect::<Vec<_>>().join(" ");
            if piece.is_empty() {
                continue;
            }
            let wide = item.x - end >= gap;
            end = item.x + item.width;
            // A form field named like its printed label repeats the label: keep the field's text only.
            if piece.starts_with(line_text.as_str()) && !line_text.is_empty() {
                line_text = piece;
                label_done = true;
                continue;
            }
            if !line_text.is_empty() {
                line_text.push_str(
                    if wide && !label_done && !line_text.ends_with([':', '：']) {
                        ": "
                    } else {
                        " "
                    },
                );
                label_done |= wide;
            }
            line_text.push_str(&piece);
        }
        if line_text.is_empty() {
            continue;
        }
        // A row in clearly larger type than most is a heading, as pdf-inspector marks it.
        let heading = row.items.len() == 1 && size(row) >= 1.3 * line;
        if let Some((bottom, after_heading)) = prev {
            let paragraph = heading || after_heading || bottom - row.top > 2.0 * line;
            text.push_str(if paragraph { "\n\n" } else { "\n" });
        }
        if heading {
            text.push_str("# ");
        }
        text.push_str(&line_text);
        prev = Some((row.bottom, heading));
    }
    Some(text)
}

/// How page markdown becomes search chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chunking {
    /// Every paragraph is a chunk; a heading is a chunk of its own (without its `#`).
    Paragraph,
    /// Every paragraph is a chunk that starts with the page's first heading ("Heading. paragraph");
    /// headings are not chunks.
    HeadingParagraph,
}

/// Cuts pages of markdown (as pdf-inspector writes them; paragraphs separated by a blank line) into chunks.
/// `pages` holds (page number, markdown).
pub fn chunks(pages: &[(u32, &str)], how: Chunking) -> Vec<Passage> {
    let mut out: Vec<Passage> = Vec::new();
    let mut section: Option<String> = None;
    for &(page, md) in pages {
        let heading = md
            .lines()
            .find(|l| l.starts_with('#'))
            .map(|l| l.trim_start_matches(['#', ' ']).trim())
            .unwrap_or("");
        for para in md.split("\n\n") {
            let text = para.trim();
            if text.is_empty() {
                continue;
            }
            let text = form_text(&strip_markup(text));
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let is_heading = para.trim_start().starts_with('#');
            let text = match how {
                Chunking::Paragraph => text.trim_start_matches('#').trim_start().to_string(),
                Chunking::HeadingParagraph if para.starts_with('#') => continue,
                Chunking::HeadingParagraph if heading.is_empty() => text.to_string(),
                Chunking::HeadingParagraph => format!("{heading}. {text}"),
            };
            match out.last_mut() {
                // A paragraph that ends in ":" introduces the next one ("… as follows:").
                Some(prev) if prev.page == page && prev.text.ends_with([':', '：']) => {
                    prev.text.push(' ');
                    prev.text.push_str(&text);
                }
                // A form label and its value can come out as paragraphs of their own (e.g. a filled form field
                // placed a little off the label's line): join them as "label: value".
                Some(prev)
                    if prev.page == page
                        && !is_heading
                        && prev.text != section.as_deref().unwrap_or("")
                        && joins_label(&prev.text, &text) =>
                {
                    prev.text.push_str(": ");
                    prev.text.push_str(&text);
                }
                _ => out.push(Passage {
                    page,
                    // A heading chunk is the section's name, not part of it.
                    section: if is_heading { None } else { section.clone() },
                    text: text.clone(),
                }),
            }
            if is_heading {
                section = Some(text);
            }
        }
    }
    out
}

/// Splits text into sentences: after `.`, `!`, `?` or `؟` followed by white space, and at list bullets;
/// abbreviations and ordinals ("Nr. 4", "1. Mai") stay inside their sentence, and a lead-in ending in ":"
/// or a question used as a heading is kept together with what follows it. Bold/underline markup is removed first.
pub fn sentences_of(text: &str) -> Vec<String> {
    let text = strip_markup(text);
    let mut out: Vec<String> = Vec::new();
    // A line break ends a sentence too (form rows, lists without bullets); a bare label line takes the line
    // after it as its value ("1.2 Date of birth: 12.03.1985").
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        // A line that repeats the line before it and adds to it (a form field named like its printed label,
        // "1.2 Date of birth" / "1.2 Date of birth: 12.03.1985") replaces it.
        if out
            .last()
            .is_some_and(|prev| line.len() > prev.len() && line.starts_with(prev.as_str()))
        {
            out.pop();
        }
        // A bare line before this one never stands alone: it is a label taking this line as its value
        // ("1.2 Date of birth: 12.03.1985"), or a heading that starts this line's first sentence.
        let join = match out.last() {
            Some(prev) if joins_label(prev, line) => Some(": "),
            Some(prev) if is_bare_label(prev) && !starts_with_bullet(line) => Some(" "),
            // A long line without a sentence end is prose wrapped onto the next line.
            Some(prev)
                if prev.chars().count() > 60
                    && !prev.ends_with(['.', '!', '?', '؟', ':', '：', ';'])
                    && !starts_with_bullet(line) =>
            {
                Some(" ")
            }
            _ => None,
        };
        for (k, sentence) in list_items(line)
            .iter()
            .flat_map(|item| split_at_stops(item))
            .enumerate()
        {
            match out.last_mut() {
                Some(prev) if k == 0 && join.is_some() => {
                    prev.push_str(join.unwrap_or(" "));
                    prev.push_str(&sentence);
                }
                // A lead-in ("as follows:") or a question used as a heading ("What is a plot?") belongs to the
                // sentence after it, which holds the answer.
                Some(lead_in) if lead_in.ends_with([':', '：', '?', '؟']) => {
                    lead_in.push(' ');
                    lead_in.push_str(&sentence);
                }
                _ => out.push(sentence),
            }
        }
    }
    out
}

/// The full-stop rule of [`sentences_of`] for one piece of text.
fn split_at_stops(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let (mut start, mut prev) = (0, None);
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c.is_whitespace() && matches!(prev, Some('.' | '!' | '?' | '؟')) {
            pieces.push(&text[start..i]);
            while chars.next_if(|&(_, w)| w.is_whitespace()).is_some() {}
            start = chars.peek().map_or(text.len(), |&(j, _)| j);
            prev = None;
        } else {
            prev = Some(c);
        }
    }
    pieces.push(&text[start..]);
    let mut out: Vec<String> = Vec::new();
    for piece in pieces {
        match out.last_mut() {
            Some(last) if ABBREVIATION.is_match(last) => {
                last.push(' ');
                last.push_str(piece);
            }
            _ => out.push(piece.to_string()),
        }
    }
    out.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The sentences of the given chunks in order, each once.
pub fn sentences(chunks: &[&Passage]) -> Vec<Passage> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for chunk in chunks {
        for text in sentences_of(&chunk.text) {
            if seen.insert(text.clone()) {
                out.push(Passage {
                    page: chunk.page,
                    text,
                    section: chunk.section.clone(),
                });
            }
        }
    }
    out
}

/// Whether a sentence reads like an amendment to an earlier clause.
pub fn is_amendment(sentence: &str) -> bool {
    AMENDMENT.is_match(sentence)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str =
        "# 3. Rent\n\nThe rent is 500 Euro. It is due monthly.\n\nParking costs 40 Euro.";

    #[test]
    fn pdf_pages_are_numbered_from_one() {
        let pages = pdf_pages(&crate::test_pdf::pdf(&[
            &["The plot rent is 120 Euro per year."],
            &["Water is included."],
        ]))
        .unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].0, 1);
        assert!(
            pages[0].1.contains("The plot rent is 120 Euro per year."),
            "{:?}",
            pages[0].1
        );
        assert_eq!(pages[1].0, 2);
        assert!(pages[1].1.contains("Water is included."));
        let c = chunks(
            &pages
                .iter()
                .map(|(n, t)| (*n, t.as_str()))
                .collect::<Vec<_>>(),
            Chunking::Paragraph,
        );
        assert_eq!(
            c.last().map(|p| (p.page, p.text.as_str())),
            Some((2, "Water is included."))
        );
    }

    #[test]
    fn a_form_grid_pdf_gives_label_value_sentences() {
        let pdf = crate::test_pdf::pdf(&[&[
            "# Garden plot application",
            "",
            "1.1 Name\tMaria Example",
            "1.2 Date of birth\t12.03.1985",
            "1.3 Plot size\t250 square metres",
        ]]);
        let pages = pdf_pages(&pdf).unwrap();
        let c = chunks(&[(1, pages[0].1.as_str())], Chunking::Paragraph);
        let sentences: Vec<String> = c.iter().flat_map(|p| sentences_of(&p.text)).collect();
        assert!(
            sentences.contains(&"1.2 Date of birth: 12.03.1985".to_string()),
            "{sentences:?}"
        );
    }

    #[test]
    fn a_form_grid_is_read_row_by_row() {
        let piece = |x: f32, y: f32, height: f32, text: &str, field: bool| Piece {
            x,
            y,
            width: 6.0 * text.chars().count() as f32,
            height,
            text: text.into(),
            field,
        };
        let mut items = vec![piece(60.0, 780.0, 16.0, "Garden plot application", false)];
        for (k, (label, value)) in [
            ("1.1 Name", "Maria Example"),
            ("1.2 Plot", "B-14"),
            ("1.3 Water wanted", "yes"),
            ("1.4 Users", "3"),
        ]
        .into_iter()
        .enumerate()
        {
            let y = 700.0 - 24.0 * k as f32;
            items.push(piece(60.0, y, 10.0, label, false));
            // A filled field: its box sits a little lower and is taller than the label's text.
            items.push(piece(300.0, y - 4.0, 16.0, value, true));
        }
        // A date typed over a fill-in line, its digits spaced out to fit the boxes.
        items.push(piece(60.0, 600.0, 10.0, "1.5 Start", false));
        items.push(piece(300.0, 600.0, 10.0, "___.___.____", false));
        items.push(piece(305.0, 599.0, 10.0, "01", false));
        items.push(piece(322.0, 599.0, 10.0, "04", false));
        items.push(piece(340.0, 599.0, 10.0, "2027", false));
        let text = form_rows(items.iter().collect()).unwrap();
        assert_eq!(
            text,
            "# Garden plot application\n\n1.1 Name: Maria Example\n1.2 Plot: B-14\n\
             1.3 Water wanted: yes\n1.4 Users: 3\n1.5 Start: 01 04 2027"
        );
        // Prose in two columns is not a form: nothing is changed.
        let prose: Vec<Piece> = (0..8)
            .flat_map(|k| {
                let y = 700.0 - 14.0 * k as f32;
                [
                    piece(
                        60.0,
                        y,
                        10.0,
                        "a long line of running text in the left column",
                        false,
                    ),
                    piece(
                        320.0,
                        y,
                        10.0,
                        "and more running text in the right column here",
                        false,
                    ),
                ]
            })
            .collect();
        assert_eq!(form_rows(prose.iter().collect()), None);
    }

    #[test]
    fn not_a_pdf_is_an_error() {
        let err = pdf_pages(b"just some text, not a PDF").unwrap_err();
        assert!(
            err.to_string().starts_with("could not read the PDF"),
            "{err}"
        );
    }

    #[test]
    fn paragraph_chunks() {
        let c = chunks(&[(2, PAGE)], Chunking::Paragraph);
        let texts: Vec<_> = c.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "3. Rent",
                "The rent is 500 Euro. It is due monthly.",
                "Parking costs 40 Euro."
            ]
        );
        assert!(c.iter().all(|p| p.page == 2));
        // Every paragraph knows its section, also on the next page; the heading itself has none.
        let c = chunks(&[(2, PAGE), (3, "Water is included.")], Chunking::Paragraph);
        let sections: Vec<_> = c.iter().map(|p| p.section.as_deref()).collect();
        assert_eq!(
            sections,
            [None, Some("3. Rent"), Some("3. Rent"), Some("3. Rent")]
        );
        let s = sentences(&[&c[1]]);
        assert_eq!(s[1].section.as_deref(), Some("3. Rent"));
    }

    #[test]
    fn heading_paragraph_chunks() {
        let c = chunks(
            &[(2, PAGE), (3, "No heading here.")],
            Chunking::HeadingParagraph,
        );
        let texts: Vec<_> = c.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "3. Rent. The rent is 500 Euro. It is due monthly.",
                "3. Rent. Parking costs 40 Euro.",
                "No heading here."
            ]
        );
    }

    #[test]
    fn sentences_keep_abbreviations_and_ordinals() {
        assert_eq!(
            sentences_of(
                "Die Miete beträgt 500 Euro. Sie ist am 1. Mai fällig!  Siehe Nr. 4 der Anlage."
            ),
            [
                "Die Miete beträgt 500 Euro.",
                "Sie ist am 1. Mai fällig!",
                "Siehe Nr. 4 der Anlage."
            ]
        );
    }

    #[test]
    fn sentences_keep_initials_and_common_abbreviations() {
        // Seen in the broad benchmark (2026-10-08): answers cut off after an initial or abbreviation.
        for (text, want) in [
            (
                "The campus was financed by Silas B. Cobb and others. It opened in 1892.",
                vec![
                    "The campus was financed by Silas B. Cobb and others.",
                    "It opened in 1892.",
                ],
            ),
            (
                "Hutton published his ideas in 1795 (Vol. 1 and 2). They were read widely.",
                vec![
                    "Hutton published his ideas in 1795 (Vol. 1 and 2).",
                    "They were read widely.",
                ],
            ),
            (
                "Two bosons (e.g. photons) behave alike, as Schuenemann et al. showed. See Fig. 3.",
                vec![
                    "Two bosons (e.g. photons) behave alike, as Schuenemann et al. showed.",
                    "See Fig. 3.",
                ],
            ),
            (
                "Es gibt Ausnahmen, z. B. für Kinder. Die Kosten betragen 5 Mio. Euro. Fertig.",
                vec![
                    "Es gibt Ausnahmen, z. B. für Kinder.",
                    "Die Kosten betragen 5 Mio. Euro.",
                    "Fertig.",
                ],
            ),
            // Words that usually end a sentence still end it.
            (
                "We sell apples, pears etc. Delivery is free.",
                vec!["We sell apples, pears etc.", "Delivery is free."],
            ),
        ] {
            assert_eq!(sentences_of(text), want, "{text}");
        }
    }

    #[test]
    fn sentences_split_after_arabic_question_mark() {
        assert_eq!(
            split_at_stops("ما هو الإيجار؟ الإيجار ٥٠٠ يورو."),
            ["ما هو الإيجار؟", "الإيجار ٥٠٠ يورو."]
        );
    }

    #[test]
    fn bold_markup_does_not_hide_a_sentence_end() {
        assert_eq!(
            split_at_stops(&strip_markup(
                "**منظور از قطعه چیست ؟** قطعه زمینی برای کاشت است."
            )),
            ["منظور از قطعه چیست ؟", "قطعه زمینی برای کاشت است."]
        );
        assert_eq!(
            sentences_of("The <u>rent</u> is **120 Euro**."),
            ["The rent is 120 Euro."]
        );
    }

    #[test]
    fn a_question_heading_keeps_its_answer() {
        assert_eq!(
            sentences_of(
                "**What is a plot?** A plot is a piece of land for growing plants. Rent is due in March."
            ),
            [
                "What is a plot? A plot is a piece of land for growing plants.",
                "Rent is due in March."
            ]
        );
    }

    #[test]
    fn a_number_moved_before_a_bullet_goes_back_to_its_item() {
        // Right-to-left list lines can come out of a PDF as "… 16 Ø متر …" for "… Ø 16 متر …".
        assert_eq!(
            sentences_of("آبیاری 3 بار در هفته است. 16 Ø متر فاصله بین ردیف ها"),
            ["آبیاری 3 بار در هفته است.", "16 متر فاصله بین ردیف ها"]
        );
        // Not in left-to-right text.
        assert_eq!(
            sentences_of("Step 16 • Close the gate"),
            ["Step 16", "Close the gate"]
        );
    }

    #[test]
    fn a_paragraph_ending_in_a_colon_joins_the_next() {
        let c = chunks(
            &[(
                1,
                "The watering order is as follows:\n\nRoses before beans.\n\nOther text.",
            )],
            Chunking::Paragraph,
        );
        let texts: Vec<_> = c.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "The watering order is as follows: Roses before beans.",
                "Other text."
            ]
        );
    }

    #[test]
    fn list_bullets_split_items() {
        // Word bullets often come out of PDFs as "Ø", "ç" or "✓" standing alone.
        assert_eq!(
            sentences_of(
                "ç Opening hours: 8 to 18 ç Water included Ø Plot size 250 m² ✓ No fires after dark"
            ),
            [
                "Opening hours: 8 to 18",
                "Water included",
                "Plot size 250 m²",
                "No fires after dark"
            ]
        );
        // The same letters inside words are not bullets.
        assert_eq!(
            sentences_of("Über Ørsted und Façade."),
            ["Über Ørsted und Façade."]
        );
    }

    #[test]
    fn a_lead_in_keeps_what_follows() {
        assert_eq!(
            sentences_of(
                "The watering order is as follows: roses before beans before grass. Other rules apply."
            ),
            [
                "The watering order is as follows: roses before beans before grass.",
                "Other rules apply."
            ]
        );
        assert_eq!(
            sentences_of("Bring along: • gloves • a spade"),
            ["Bring along: gloves", "a spade"]
        );
    }

    #[test]
    fn sentences_are_not_repeated() {
        let a = Passage {
            page: 1,
            text: "Same. First.".into(),
            section: None,
        };
        let b = Passage {
            page: 2,
            text: "Same. Second.".into(),
            section: None,
        };
        let s = sentences(&[&a, &b]);
        let got: Vec<_> = s.iter().map(|p| (p.page, p.text.as_str())).collect();
        assert_eq!(got, [(1, "Same."), (1, "First."), (2, "Second.")]);
    }

    #[test]
    fn amendments() {
        assert!(is_amendment(
            "Amendment No. 1: with effect from 1 May the rent is 450 Euro."
        ));
        assert!(is_amendment("Nachtrag 2: Die Miete wird erhöht."));
        assert!(!is_amendment("The rent is 500 Euro."));
    }

    #[test]
    fn form_rows_become_label_value_sentences() {
        // A filled form: labels with fill-in lines, a label whose value is on the next line, a field named
        // like its label, and a numbered label list that must not chain.
        let form = "1.1 Name: ______ Maria Example\n1.2 Date of birth\n12.03.1985\n\
                    1.3 Plot\n1.3 Plot: B-14\n2.1 Water\n2.2 Power";
        let c = chunks(&[(1, form)], Chunking::Paragraph);
        assert_eq!(
            sentences_of(&c[0].text),
            [
                "1.1 Name: Maria Example",
                "1.2 Date of birth: 12.03.1985",
                "1.3 Plot: B-14",
                "2.1 Water 2.2 Power"
            ]
        );
        // A label and its value in paragraphs of their own (a form field placed off the label's line).
        let c = chunks(
            &[(1, "2.4 Number of users\n\nTextfeld8: 3")],
            Chunking::Paragraph,
        );
        assert_eq!(c[0].text, "2.4 Number of users: Textfeld8: 3");
    }

    #[test]
    fn tables_become_sentences() {
        let two = "|1.1 Name|Maria Example|\n|---|---|\n|1.2 Plot|B-14|";
        assert_eq!(form_text(two), "1.1 Name: Maria Example\n1.2 Plot: B-14");
        let three = "|Plan|Water|Fee|\n|---|---|---|\n|Small|200 l|40 Euro|";
        assert_eq!(form_text(three), "Plan: Small, Water: 200 l, Fee: 40 Euro");
    }

    #[test]
    fn headings_and_wrapped_prose_stay_with_their_text() {
        // A numbered heading before a long paragraph starts its first sentence, as before.
        let long = "Die Pacht ist jährlich im Voraus zu zahlen, spätestens bis zum ersten Werktag im März.";
        assert_eq!(
            sentences_of(&format!("4. Pacht\n{long}")),
            [format!("4. Pacht {long}")]
        );
        // A long line without a sentence end is wrapped prose.
        assert_eq!(
            sentences_of(
                "Wer die Hecke nicht rechtzeitig schneidet, erhält vom Vorstand des Vereins eine\nAbmahnung. Danke."
            ),
            [
                "Wer die Hecke nicht rechtzeitig schneidet, erhält vom Vorstand des Vereins eine Abmahnung.",
                "Danke."
            ]
        );
        // A word starting with a bullet letter is not a bullet.
        assert_eq!(sentences_of("Anzahl\nvierzig"), ["Anzahl: vierzig"]);
        assert_eq!(
            sentences_of("Mitbringen\n• Handschuhe"),
            ["Mitbringen", "Handschuhe"]
        );
    }
}

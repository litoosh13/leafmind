//! Text from Indra's recognizers: their vocabularies, and right-to-left lines turned from the order the
//! recognizer reads them (left to right on the page) into logical order (indra_ocr/rtl.py).

use unicode_bidi::{BidiClass, bidi_class};
use unicode_normalization::UnicodeNormalization;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// A recognizer's tokens: class 0 is the CTC blank, class i + 1 is `tokens[i]` (a character, or a LaTeX
/// command in the Latin recognizer).
pub(crate) struct Vocab {
    tokens: Vec<String>,
    has_commands: bool,
}

impl Vocab {
    /// From meta.json's `{"blank": 0, "chars": [...]}`.
    pub fn from_json(s: &str) -> Result<Vocab, String> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if v["blank"].as_i64().unwrap_or(0) != 0 {
            return Err("only blank = 0 vocabularies are supported".into());
        }
        let tokens: Vec<String> = v["chars"]
            .as_array()
            .ok_or("vocabulary without chars")?
            .iter()
            .map(|c| c.as_str().unwrap_or_default().to_string())
            .collect();
        Ok(Vocab {
            has_commands: tokens.iter().any(|t| t.chars().count() > 1),
            tokens,
        })
    }

    pub fn token(&self, id: i64) -> &str {
        match id {
            1.. if (id as usize) <= self.tokens.len() => &self.tokens[id as usize - 1],
            _ => "",
        }
    }

    /// The text of tokens; a LaTeX command before a letter keeps a space ("\alpha x").
    pub fn decode_tokens(&self, toks: &[&str]) -> String {
        let mut out = String::new();
        for (k, t) in toks.iter().enumerate() {
            out.push_str(t);
            let next = toks.get(k + 1).and_then(|n| n.chars().next());
            if self.has_commands
                && t.len() > 2
                && t.starts_with('\\')
                && t[1..].chars().all(char::is_alphabetic)
                && next.is_some_and(char::is_alphabetic)
            {
                out.push(' ');
            }
        }
        out
    }
}

pub(crate) fn is_rtl(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::R | BidiClass::AL)
}

fn is_ltr(c: char) -> bool {
    // Digits (European and Arabic-Indic) read left to right inside right-to-left text.
    matches!(bidi_class(c), BidiClass::L | BidiClass::EN | BidiClass::AN)
}

pub(crate) fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// More right-to-left letters than left-to-right ones.
fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let l = text
        .chars()
        .filter(|&c| bidi_class(c) == BidiClass::L)
        .count();
    r > l
}

fn is_mark(c: char) -> bool {
    matches!(
        c.general_category(),
        GeneralCategory::NonspacingMark | GeneralCategory::EnclosingMark
    )
}

/// Base characters with their combining marks.
fn clusters(text: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for c in text.chars() {
        match out.last_mut() {
            Some(last) if is_mark(c) => last.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    R,
    L,
    N,
}

fn kind(cluster: &str) -> Kind {
    let c = cluster.chars().next().unwrap_or(' ');
    if is_rtl(c) {
        Kind::R
    } else if is_ltr(c) {
        Kind::L
    } else {
        Kind::N
    }
}

/// [start, end) spans from a `strong` cluster to the last `strong` one before a cluster of kind `stop`.
fn runs(kinds: &[Kind], strong: Kind, stop: Kind) -> Vec<(usize, usize)> {
    let (mut spans, mut start, mut last) = (vec![], None, 0);
    for i in 0..=kinds.len() {
        match kinds.get(i) {
            Some(&k) if k == strong => {
                start.get_or_insert(i);
                last = i;
            }
            Some(&k) if k != stop => {}
            _ => {
                if let Some(s) = start.take() {
                    spans.push((s, last + 1));
                }
            }
        }
    }
    spans
}

fn mirror(cluster: &str) -> String {
    const PAIRS: [(char, char); 12] = [
        ('(', ')'),
        (')', '('),
        ('[', ']'),
        (']', '['),
        ('{', '}'),
        ('}', '{'),
        ('<', '>'),
        ('>', '<'),
        ('«', '»'),
        ('»', '«'),
        ('‹', '›'),
        ('›', '‹'),
    ];
    let mut chars = cluster.chars();
    let first = chars.next().unwrap_or(' ');
    let first = PAIRS.iter().find(|p| p.0 == first).map_or(first, |p| p.1);
    std::iter::once(first).chain(chars).collect()
}

/// Visual order to logical order (and back: the same call). `base_rtl`: the paragraph direction when known.
pub(crate) fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let cls = clusters(text);
    let kinds: Vec<Kind> = cls.iter().map(|c| kind(c)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        // Units: left-to-right runs kept whole, every other cluster alone; the units reversed.
        let mut units: Vec<String> = vec![];
        let mut i = 0;
        let mut spans = runs(&kinds, Kind::L, Kind::R);
        spans.push((cls.len(), cls.len()));
        for (a, b) in spans {
            units.extend(cls[i..a].iter().map(|c| mirror(c)));
            if a < b {
                units.push(cls[a..b].concat());
            }
            i = b;
        }
        return units.into_iter().rev().collect();
    }
    // Left-to-right base: each right-to-left run reversed in place.
    let mut out = cls.clone();
    for (a, b) in runs(&kinds, Kind::R, Kind::L) {
        for (k, c) in cls[a..b].iter().rev().enumerate() {
            out[a + k] = mirror(c);
        }
    }
    out.concat()
}

/// The text of a group recognizer's characters (visual order): white space collapsed, NFC, right-to-left
/// runs in logical order.
pub(crate) fn group_text(chars: &str, page_rtl: bool) -> String {
    let text: String = chars
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .nfc()
        .collect();
    if has_rtl(&text) {
        flip(&text, page_rtl.then_some(true))
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_to_left_lines_come_out_in_logical_order() {
        // The recognizer reads "سلام دنیا" left to right on the page: "ایند مالس".
        assert_eq!(flip("ایند مالس", None), "سلام دنیا");
        // Digits and Latin words keep their own order inside the line.
        assert_eq!(flip("2026 لاس", None), "سال 2026");
        // Brackets turn round at right-to-left level: as drawn, "(ب)" reads "(ب)" from the left too.
        assert_eq!(flip("(ب)", None), "(ب)");
        assert_eq!(flip("(ب) ج", None), "ج (ب)");
        // A left-to-right line keeps its order; only the Persian run turns.
        assert_eq!(flip("Name: ینام ok", None), "Name: مانی ok");
        // Its own inverse.
        assert_eq!(
            flip(&flip("متن (آزمایشی) 12", None), None),
            "متن (آزمایشی) 12"
        );
        assert_eq!(group_text("  ایند   مالس ", false), "سلام دنیا");
    }
}

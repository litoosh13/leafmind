//! Keyword search (BM25), merging rankings, and choosing the answer sentences from scores.

use crate::{Passage, text::is_amendment};
use std::collections::HashMap;

/// Question words and fillers (English and German) that say nothing about the answer.
const STOP_WORDS: &[&str] = &[
    "the", "is", "was", "what", "who", "how", "of", "and", "to", "in", "does", "can", "which",
    "when", "are", "for", "a", "an", "der", "das", "und", "ist", "wie", "welche", "welcher",
    "darf", "hoch", "kostet", "ein", "eine", "wer", "wann", "muss", "nicht", "mit", "dem", "den",
    "des", "zu", "auf", "für", "von", "beträgt", "wird", "sind", "oder", "gefunden", "be", "it",
    "this", "that", "by", "with", "at", "per", "from", "as", "or", "on", "any", "all", "shall",
    "must",
];

/// Cuts a common English ending so "payments" matches "payment".
fn stem(word: &str) -> &str {
    let n = word.chars().count();
    for suffix in ["ting", "ing", "ies", "es", "ed", "s"] {
        if n > suffix.len() + 3 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

/// Arabic vowel marks and Quranic annotation signs. Questions are usually typed without them, so they are
/// dropped before matching ("شهرياً" matches "شهريا", and a shadda no longer splits a word).
fn is_arabic_mark(c: char) -> bool {
    matches!(c, '\u{064B}'..='\u{065F}' | '\u{0670}' | '\u{06D6}'..='\u{06ED}')
}

/// Lower-case words (letters, digits, `_`) longer than one character, minus stop words, stemmed.
fn words(text: &str) -> Vec<String> {
    let text: String = text.chars().filter(|&c| !is_arabic_mark(c)).collect();
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().count() > 1 && !STOP_WORDS.contains(w))
        .map(|w| stem(w).to_string())
        .collect()
}

/// Okapi BM25 keyword index over a document's chunks (k1 = 1.5, b = 0.75).
#[derive(Clone, Debug)]
pub struct Bm25 {
    counts: Vec<HashMap<String, f64>>,
    lengths: Vec<f64>,
    average: f64,
    idf: HashMap<String, f64>,
}

impl Bm25 {
    pub fn new(texts: &[&str]) -> Self {
        let counts: Vec<HashMap<String, f64>> = texts
            .iter()
            .map(|t| {
                let mut m = HashMap::new();
                for w in words(t) {
                    *m.entry(w).or_insert(0.0) += 1.0;
                }
                m
            })
            .collect();
        let lengths: Vec<f64> = counts.iter().map(|c| c.values().sum()).collect();
        let average = lengths.iter().sum::<f64>() / lengths.len().max(1) as f64;
        let mut df: HashMap<String, f64> = HashMap::new();
        for c in &counts {
            for w in c.keys() {
                *df.entry(w.clone()).or_insert(0.0) += 1.0;
            }
        }
        let n = counts.len() as f64;
        let idf = df
            .into_iter()
            .map(|(w, f)| (w, (1.0 + (n - f + 0.5) / (f + 0.5)).ln()))
            .collect();
        Bm25 {
            counts,
            lengths,
            average,
            idf,
        }
    }

    /// One score per chunk; 0 where no question word occurs.
    pub fn scores(&self, query: &str) -> Vec<f64> {
        let (k1, b) = (1.5, 0.75);
        let q = words(query);
        self.counts
            .iter()
            .zip(&self.lengths)
            .map(|(c, len)| {
                q.iter()
                    .filter_map(|w| {
                        let f = c.get(w)?;
                        Some(
                            self.idf[w] * f * (k1 + 1.0)
                                / (f + k1 * (1.0 - b + b * len / self.average)),
                        )
                    })
                    .sum()
            })
            .collect()
    }
}

/// Indices from the highest score to the lowest; equal scores keep their order.
pub fn order(scores: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..scores.len()).collect();
    o.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    o
}

/// Reciprocal rank fusion (k = 60): merges several score lists over the same items into one, using only
/// each item's rank in each list.
pub fn rrf(score_lists: &[&[f64]]) -> Vec<f64> {
    let n = score_lists.first().map_or(0, |s| s.len());
    let mut fused = vec![0.0; n];
    for scores in score_lists {
        for (rank, i) in order(scores).into_iter().enumerate() {
            fused[i] += 1.0 / (60.0 + rank as f64);
        }
    }
    fused
}

/// The answer: the best-scoring sentence, then up to two amendment sentences that score at least 0.3 × the
/// best (an amendment often scores lower than the clause it changes). Returns indices into `sentences`.
pub fn pick(scores: &[f64], sentences: &[Passage]) -> Vec<usize> {
    let Some(&best) = order(scores).first() else {
        return vec![];
    };
    let amendments = (0..sentences.len())
        .filter(|&i| {
            i != best && scores[i] >= 0.3 * scores[best] && is_amendment(&sentences[i].text)
        })
        .take(2);
    std::iter::once(best).chain(amendments).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_words_stems_and_short_words_go() {
        assert_eq!(
            words("How much are the monthly Payments? A 5% fee."),
            ["much", "monthly", "payment", "fee"]
        );
    }

    #[test]
    fn arabic_vowel_marks_are_ignored() {
        assert_eq!(words("يدفع شهرياً المؤمَّن"), words("يدفع شهريا المؤمن"));
        assert_eq!(words("المؤمَّن"), ["المؤمن"]);
    }

    #[test]
    fn bm25_finds_the_matching_chunk() {
        let bm = Bm25::new(&[
            "The rent is 500 Euro per month.",
            "Parking costs 40 Euro.",
            "The garden is shared.",
        ]);
        let s = bm.scores("How much is the rent?");
        assert_eq!(order(&s)[0], 0);
        assert!(s[1] == 0.0 && s[2] == 0.0);
        assert!(bm.scores("What is the?").iter().all(|&x| x == 0.0));
    }

    #[test]
    fn order_is_stable_for_ties() {
        assert_eq!(order(&[0.5, 0.9, 0.5, 0.1]), [1, 0, 2, 3]);
    }

    #[test]
    fn rrf_rewards_agreement() {
        let a = [0.9, 0.8, 0.1];
        let b = [0.2, 0.7, 0.6];
        assert_eq!(order(&rrf(&[&a, &b])), [1, 0, 2]);
    }

    #[test]
    fn pick_adds_strong_amendments_only() {
        let s = |t: &str| Passage {
            page: 1,
            text: t.into(),
            section: None,
        };
        let sents = [
            s("The rent is 500 Euro."),
            s("Amendment 1: with effect from May the rent is 450 Euro."),
            s("Amendment 2: the garden is shared."),
            s("Parking costs 40 Euro."),
        ];
        assert_eq!(pick(&[0.9, 0.5, 0.1, 0.2], &sents), [0, 1]);
        assert_eq!(pick(&[], &[]), Vec::<usize>::new());
    }
}

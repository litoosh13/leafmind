//! The question-answering flow without the models: the document, search, the candidate sentences and the
//! decision to answer or withhold. Shared by the native engine (models on ONNX Runtime) and the browser build
//! (models run by the page), so both answer the same way.

use crate::Passage;
use crate::language::Language;
use crate::search::{self, Bm25};
use crate::text::{self, Chunking};

/// How many best chunks are split into sentences for the reranker.
pub(crate) const TOP_CHUNKS: usize = 8;

/// A document is a form when at least this share of its sentences are short form rows ("label: value", value
/// ≤ 60 characters). On a broad public benchmark (14 sets) no document reached 0.4 (long reports up to 0.3, where
/// the forms rule made wrong answers); invented forms are around 0.75.
const FORM_SHARE: f64 = 0.4;

/// In a form, a row whose label the question names completely (every word but fillers) is answered from this
/// reranker score on, below `min_confidence`: the reranker scores form rows low (0.4–0.5), all-caps labels lower.
/// 0.4 added 13 right answers on invented scanned forms and none wrong on any of our sets; 0.3 made wrong ones.
const NAMED_FLOOR: f64 = 0.4;

#[derive(Clone, Copy, Debug)]
pub struct QaOptions {
    /// CPU threads per model; 0 = half the logical CPUs (about the physical cores). Native only.
    pub threads: usize,
    /// Below this reranker score the answer is withheld ([`Answer::NotFound`]); in a form, a row whose label the
    /// question names completely needs only 0.4.
    pub min_confidence: f64,
    /// The answer is also withheld when another sentence scores almost as high: the answer's score must
    /// beat the best sentence outside the answer by at least this much. 0.15 cut the wrong answers shown on a
    /// broad public benchmark (14 sets, 4 languages) from 666 to 98 and kept 1,433 of 2,083 right ones; lower it
    /// (0.1, 0.05, 0) for more answers and more wrong ones. Not in a form for a row ("label: value") whose label
    /// the question names: form rows score close to each other, and the label says which one is asked for.
    pub min_gap: f64,
    /// How the document is cut into search chunks.
    pub chunking: Chunking,
}

impl Default for QaOptions {
    fn default() -> Self {
        QaOptions {
            threads: 0,
            min_confidence: 0.5,
            min_gap: 0.15,
            chunking: Chunking::Paragraph,
        }
    }
}

/// A document ready for questions. Plain data, so an app can keep it while the document is open.
#[derive(Clone, Debug)]
pub struct Document {
    pub(crate) chunks: Vec<Passage>,
    bm25: Bm25,
    pub(crate) vectors: Vec<Vec<f32>>,
    pub(crate) language: Option<Language>,
    /// Whether the document is a form ([`FORM_SHARE`]): only then the forms rule of [`decide`] applies.
    pub(crate) form: bool,
}

impl Document {
    /// From its chunks, their unit-length embeddings (same order) and its language.
    pub(crate) fn new(
        chunks: Vec<Passage>,
        vectors: Vec<Vec<f32>>,
        language: Option<Language>,
    ) -> Self {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let sentences: Vec<String> = texts.iter().flat_map(|t| text::sentences_of(t)).collect();
        let rows = sentences
            .iter()
            .filter(|s| {
                text::is_label_value(s)
                    && s.split_once([':', '：'])
                        .is_some_and(|(_, v)| v.trim().chars().count() <= 60)
            })
            .count();
        Document {
            form: rows as f64 >= FORM_SHARE * sentences.len().max(1) as f64,
            bm25: Bm25::new(&texts),
            vectors,
            chunks,
            language,
        }
    }

    /// The search chunks, in document order.
    pub fn chunks(&self) -> &[Passage] {
        &self.chunks
    }

    /// The document's language, if it could be told (questions must be asked in it).
    pub fn language(&self) -> Option<Language> {
        self.language
    }
}

/// The result of [`crate::QaEngine::ask`].
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// The sentence that answers the question (then amendments that change it), exactly as in the document.
    Found {
        sentences: Vec<Passage>,
        confidence: f64,
    },
    /// No sentence scored high enough, or another sentence scored almost as high: the answer is withheld rather
    /// than guessed.
    NotFound { best_confidence: f64 },
    /// The question is clearly in another language than the document: ask again in the document's language.
    WrongLanguage {
        document: Language,
        question: Language,
    },
}

/// The vector scaled to unit length (the embedder's output, so that dot products are cosine similarities).
pub(crate) fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// The reranker's logit as a probability (0..1) that the sentence answers the question.
pub(crate) fn probability(logit: f32) -> f64 {
    1.0 / (1.0 + (-logit as f64).exp())
}

/// The sentences the reranker judges: those of the best chunks for the question, found by embedding similarity
/// (`question_vector`, unit length) and keywords, merged by reciprocal rank fusion.
pub(crate) fn candidates(doc: &Document, question: &str, question_vector: &[f32]) -> Vec<Passage> {
    let similarity: Vec<f64> = doc
        .vectors
        .iter()
        .map(|v| {
            v.iter()
                .zip(question_vector)
                .map(|(a, b)| a * b)
                .sum::<f32>() as f64
        })
        .collect();
    let keyword = doc.bm25.scores(question);
    // Keyword ranks only count when some question word occurs at all.
    let fused = if keyword.iter().any(|&s| s > 0.0) {
        search::rrf(&[&similarity, &keyword])
    } else {
        search::rrf(&[&similarity])
    };
    // RESEARCH-TMP LM_CANDDUMP: the 30 best chunks in order (removed after the long-document study).
    if let Ok(f) = std::env::var("LM_CANDDUMP") {
        use std::io::Write;
        let chunks: Vec<&str> = search::order(&fused)
            .into_iter()
            .take(30)
            .map(|i| doc.chunks[i].text.as_str())
            .collect();
        let pages: Vec<u32> = search::order(&fused)
            .into_iter()
            .take(30)
            .map(|i| doc.chunks[i].page)
            .collect();
        let line = format!(
            "{{\"question\":{:?},\"chunks\":{:?},\"pages\":{:?},\"n_chunks\":{}}}\n",
            question,
            chunks,
            pages,
            doc.chunks.len()
        );
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{f}.chunks"))
            .unwrap()
            .write_all(line.as_bytes())
            .unwrap();
    }
    // RESEARCH-TMP LM_TOP / LM_TOP_LONG: chunks for the reranker (LM_TOP_LONG when the document has > 40 chunks).
    let n_top = match (
        std::env::var("LM_TOP_LONG")
            .ok()
            .and_then(|v| v.parse().ok()),
        doc.chunks.len() > 40,
    ) {
        (Some(n), true) => n,
        _ => TOP_CHUNKS,
    };
    let top: Vec<&Passage> = search::order(&fused)
        .into_iter()
        .take(n_top)
        .map(|i| &doc.chunks[i])
        .collect();
    text::sentences(&top)
}

/// The answer from the reranker's `scores` of the `sentences`: the best one and its amendments, unless it scores
/// below `min_confidence` or another sentence scores almost as high (`min_gap`). In the second case, in a form
/// (`form`, see [`Document`]), a form row whose label the question names is still the answer ([`form_row`]).
pub(crate) fn decide(
    options: &QaOptions,
    form: bool,
    question: &str,
    scores: &[f64],
    sentences: &[Passage],
) -> Answer {
    // RESEARCH-TMP LM_CANDDUMP: candidates with scores.
    if let Ok(f) = std::env::var("LM_CANDDUMP") {
        use std::io::Write;
        let c: Vec<String> = sentences
            .iter()
            .zip(scores)
            .map(|(s, sc)| format!("[{:?},{},{}]", s.text, s.page, sc))
            .collect();
        let line = format!(
            "{{\"question\":{:?},\"cands\":[{}]}}\n",
            question,
            c.join(",")
        );
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{f}.cands"))
            .unwrap()
            .write_all(line.as_bytes())
            .unwrap();
    }
    let picked = search::pick(scores, sentences);
    let confidence = picked.first().map_or(0.0, |&i| scores[i]);
    let clear = search::gap(scores, &picked, sentences) >= options.min_gap;
    if !(confidence >= options.min_confidence && clear)
        && form
        && let Some(i) = form_row(question, scores, sentences, options.min_confidence)
    {
        return Answer::Found {
            sentences: vec![sentences[i].clone()],
            confidence: scores[i],
        };
    }
    if confidence >= options.min_confidence && clear {
        Answer::Found {
            sentences: picked.iter().map(|&i| sentences[i].clone()).collect(),
            confidence,
        }
    } else {
        Answer::NotFound {
            best_confidence: confidence,
        }
    }
}

/// Question words and fillers (EN/DE/FA/AR) that ask rather than name what is asked; they never match a label.
const ASKING: &[&str] = &[
    "what", "which", "when", "where", "who", "whom", "whose", "how", "much", "many", "does", "did",
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "from", "will", "should",
    "have", "has", "can", "wie", "wann", "wer", "welche", "welcher", "welches", "lautet", "ist",
    "sind", "der", "die", "das", "den", "dem", "des", "und", "hat", "soll", "wird", "viel",
    "viele", "für", "von", "mit", "چه", "چیست", "کدام", "چند", "است", "این", "متى", "أين", "كم",
    "على",
];

/// The words of `text` with at least 3 characters and a letter, in lower case.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3 && w.chars().any(char::is_alphabetic))
        .map(str::to_lowercase)
        .collect()
}

/// Two words match when they start alike ("applicant" / "applicant's", "start" / "starts"): the first 5
/// characters, or all of the shorter one.
fn same_word(a: &str, b: &str) -> bool {
    let n = a.chars().count().min(b.chars().count()).min(5);
    a.chars().take(n).eq(b.chars().take(n))
}

/// The best-scoring form row ("label: value") whose label the question names: at least half of the label's
/// words (fillers like "of", "der" aside) are in the question, and its score reaches `min_confidence`, or
/// [`NAMED_FLOOR`] when the question names all of them. Form rows score close to each other with the reranker,
/// so the gap rule withholds them; the label says which row is asked for.
fn form_row(
    question: &str,
    scores: &[f64],
    sentences: &[Passage],
    min_confidence: f64,
) -> Option<usize> {
    let asked: Vec<String> = words(question)
        .into_iter()
        .filter(|w| !ASKING.contains(&w.as_str()))
        .collect();
    let mut best: Option<usize> = None;
    for (i, s) in sentences.iter().enumerate() {
        let Some((label, _)) = s.text.split_once([':', '：']) else {
            continue;
        };
        let label: Vec<String> = words(label)
            .into_iter()
            .filter(|w| !ASKING.contains(&w.as_str()))
            .collect();
        if !text::is_label_value(&s.text) || label.is_empty() {
            continue;
        }
        let named = label
            .iter()
            .filter(|l| asked.iter().any(|q| same_word(l, q)))
            .count();
        let floor = if named == label.len() {
            min_confidence.min(NAMED_FLOOR)
        } else {
            min_confidence
        };
        if scores[i] < floor {
            continue;
        }
        if 2 * named >= label.len() && best.is_none_or(|b| scores[i] > scores[b]) {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(texts: &[&str]) -> Vec<Passage> {
        texts
            .iter()
            .map(|t| Passage {
                page: 1,
                text: t.to_string(),
                section: None,
            })
            .collect()
    }

    #[test]
    fn a_form_row_named_by_the_question_skips_the_gap_rule() {
        let options = QaOptions::default();
        let s = rows(&[
            "2.1 Requested plot number: B-14",
            "2.2 Plot size: 250",
            "Notes: none",
        ]);
        let scores = [0.57, 0.58, 0.55];
        // Close scores: the gap rule alone withholds; the label "Plot size" is named by the question.
        assert!(matches!(
            decide(&options, true, "Who signs?", &scores, &s),
            Answer::NotFound { .. }
        ));
        match decide(&options, true, "How big is the plot size?", &scores, &s) {
            Answer::Found { sentences, .. } => assert_eq!(sentences[0].text, "2.2 Plot size: 250"),
            other => panic!("{other:?}"),
        }
        // A completely named label from 0.4 on, below min_confidence.
        assert!(matches!(
            decide(&options, true, "What is the plot size?", &[0.3, 0.42, 0.35], &s),
            Answer::Found { confidence, .. } if confidence == 0.42
        ));
        assert!(matches!(
            decide(
                &options,
                true,
                "What is the plot size?",
                &[0.3, 0.38, 0.35],
                &s
            ),
            Answer::NotFound { .. }
        ));
        // Labels in capitals are scored in lower case.
        assert_eq!(
            scoring_text("CLUB ADDRESS: 3 Orchard Road"),
            "club address: 3 Orchard Road"
        );
        assert_eq!(
            scoring_text("Club address: 3 Orchard Road"),
            "Club address: 3 Orchard Road"
        );
        assert_eq!(scoring_text("THE RENT IS DUE."), "THE RENT IS DUE.");
        // Not in a document that is no form.
        assert!(matches!(
            decide(&options, false, "How big is the plot size?", &scores, &s),
            Answer::NotFound { .. }
        ));
        // Of two named rows the better-scoring one; a half-named label never below min_confidence.
        assert!(matches!(
            decide(&options, true, "Requested plot size?", &[0.57, 0.45, 0.55], &s),
            Answer::Found { confidence, .. } if confidence == 0.57
        ));
        assert!(matches!(
            decide(&options, true, "Requested plot?", &[0.45, 0.3, 0.42], &s),
            Answer::NotFound { .. }
        ));
    }
}

#[cfg(test)]
mod form_tests {
    use super::*;

    #[test]
    fn a_document_is_a_form_by_its_share_of_form_rows() {
        let chunk = |text: &str| Passage {
            page: 1,
            text: text.into(),
            section: None,
        };
        let doc = |texts: &[&str]| {
            let chunks: Vec<Passage> = texts.iter().map(|t| chunk(t)).collect();
            let n = chunks.len();
            Document::new(chunks, vec![vec![1.0]; n], None).form
        };
        assert!(doc(&[
            "Name: Maria Example\nPlot: B-14\nSize: 250",
            "Signature"
        ]));
        assert!(!doc(&[
            "The lease runs for two years. Rent is due monthly. Parking is extra.",
            "Note: the garden closes at dusk."
        ]));
    }
}

/// The text the reranker scores: a form row's label in capitals ("CLUB ADDRESS: …") is scored in lower case
/// ("club address: …"); the reranker scores capitals much lower (0.30 against 0.54 for the same row).
pub(crate) fn scoring_text(text: &str) -> std::borrow::Cow<'_, str> {
    match text.split_once([':', '：']) {
        Some((label, _))
            if text::is_label_value(text)
                && label.chars().filter(|c| c.is_alphabetic()).count() >= 2
                && !label.chars().any(char::is_lowercase) =>
        {
            format!("{}{}", label.to_lowercase(), &text[label.len()..]).into()
        }
        _ => text.into(),
    }
}

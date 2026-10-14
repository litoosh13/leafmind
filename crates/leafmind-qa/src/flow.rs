//! The question-answering flow without the models: the document, search, the candidate sentences and the
//! decision to answer or withhold. Shared by the native engine (models on ONNX Runtime) and the browser build
//! (models run by the page), so both answer the same way.

use crate::Passage;
use crate::language::Language;
use crate::search::{self, Bm25};
use crate::text::{self, Chunking};

/// How many best chunks are split into sentences for the reranker.
pub(crate) const TOP_CHUNKS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct QaOptions {
    /// CPU threads per model; 0 = half the logical CPUs (about the physical cores). Native only.
    pub threads: usize,
    /// Below this reranker score the answer is withheld ([`Answer::NotFound`]).
    pub min_confidence: f64,
    /// The answer is also withheld when another sentence scores almost as high: the answer's score must
    /// beat the best sentence outside the answer by at least this much. 0.15 cut the wrong answers shown on a
    /// broad public benchmark (14 sets, 4 languages) from 666 to 98 and kept 1,433 of 2,083 right ones; lower it
    /// (0.1, 0.05, 0) for more answers and more wrong ones.
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
}

impl Document {
    /// From its chunks, their unit-length embeddings (same order) and its language.
    pub(crate) fn new(
        chunks: Vec<Passage>,
        vectors: Vec<Vec<f32>>,
        language: Option<Language>,
    ) -> Self {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        Document {
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
    let top: Vec<&Passage> = search::order(&fused)
        .into_iter()
        .take(TOP_CHUNKS)
        .map(|i| &doc.chunks[i])
        .collect();
    text::sentences(&top)
}

/// The answer from the reranker's `scores` of the `sentences`: the best one and its amendments, unless it scores
/// below `min_confidence` or another sentence scores almost as high (`min_gap`).
pub(crate) fn decide(options: &QaOptions, scores: &[f64], sentences: &[Passage]) -> Answer {
    let picked = search::pick(scores, sentences);
    let confidence = picked.first().map_or(0.0, |&i| scores[i]);
    let clear = search::gap(scores, &picked, sentences) >= options.min_gap;
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

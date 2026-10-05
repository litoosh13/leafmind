//! Answers a question about a document by **picking** the sentence that answers it — it never writes an
//! answer of its own, so what it shows is always the document's own words, numbers included.
//!
//! The pipeline (in progress): PDF text → paragraphs ([`text::chunks`]) → keyword search ([`search::Bm25`]) plus
//! embedding search, merged by reciprocal rank fusion ([`search::rrf`]) → the best paragraphs' sentences
//! ([`text::sentences`]) → a reranker scores each sentence → [`search::pick`] keeps the best one (and amendments
//! that change it). If the best score is too low the answer is withheld instead of guessed.
//!
//! ```no_run
//! use leafmind_qa::{Answer, QaEngine, QaModels, QaOptions};
//! let engine = QaEngine::load(
//!     &QaModels {
//!         onnxruntime: "lib/libonnxruntime.dylib".into(),
//!         embedder: "models/gte-embed".into(),
//!         reranker: "models/gte-reranker".into(),
//!         accurate_reranker: Some("models/qwen3-reranker".into()), // optional, for ask_accurate
//!     },
//!     QaOptions::default(),
//! )?;
//! let doc = engine.index_pdf(&std::fs::read("contract.pdf")?)?;
//! // engine.ask_accurate(…) instead: slower, finds more answers (see QaEngine::ask_accurate).
//! match engine.ask(&doc, "How much is the rent?")? {
//!     Answer::Found { sentences, confidence } => println!("{sentences:?} ({confidence:.2})"),
//!     Answer::NotFound { .. } => println!("not found in the document"),
//!     Answer::WrongLanguage { document, .. } => println!("please ask in {document:?}"),
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;

mod accurate;
mod engine;
pub mod language;
#[cfg(test)]
mod model_tests;
pub mod search;
#[cfg(test)]
mod test_pdf;
pub mod text;

pub use engine::{Answer, Document, QaEngine, QaModels, QaOptions};
pub use language::Language;

/// What can go wrong.
#[derive(Debug)]
pub enum Error {
    /// The bytes are not a PDF that could be read (the message says why).
    Pdf(String),
    /// ONNX Runtime or a model could not be loaded or run.
    Model(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Pdf(why) => write!(f, "could not read the PDF: {why}"),
            Error::Model(why) => write!(f, "model error: {why}"),
        }
    }
}

impl std::error::Error for Error {}

/// A piece of the document's text with the page it is on (1-based): a paragraph or a sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Passage {
    pub page: u32,
    pub text: String,
    /// The heading of the section it is in (the latest heading above it), if the document has headings.
    pub section: Option<String>,
}

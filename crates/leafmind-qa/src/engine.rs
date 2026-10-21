//! The native engine: gte-multilingual-base embeds chunks and questions, gte-multilingual-reranker-base scores
//! (question, sentence) pairs. Both run on ONNX Runtime, which the app ships and names by path. The flow around
//! the models is in `flow`, shared with the browser build.

use crate::flow::{self, Answer, Document, QaOptions};
use crate::language::LanguageCheck;
use crate::saved;
use crate::text;
use crate::{Error, Passage};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

/// Longest input the models see, in tokens.
const MAX_TOKENS: usize = 512;

/// Where the app keeps ONNX Runtime and the two models. Each model folder holds `model_int8.onnx` and
/// `tokenizer.json` (see THIRD_PARTY.md for the sources).
#[derive(Clone, Debug)]
pub struct QaModels {
    /// The ONNX Runtime library: `onnxruntime.dll`, `libonnxruntime.so` or `libonnxruntime.dylib` (1.30).
    pub onnxruntime: PathBuf,
    /// gte-multilingual-base (search).
    pub embedder: PathBuf,
    /// gte-multilingual-reranker-base (picks the answer sentence).
    pub reranker: PathBuf,
}

pub(crate) fn model_error(e: impl std::fmt::Display) -> Error {
    Error::Model(e.to_string())
}

/// One ONNX model with its tokenizer. The session is behind a lock because running it needs `&mut`.
struct Model {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Model {
    fn load(dir: &std::path::Path, threads: usize) -> Result<Self, Error> {
        let session = Session::builder()
            .map_err(model_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_error)?
            .with_intra_threads(threads)
            .map_err(model_error)?
            .with_inter_threads(1)
            .map_err(model_error)?
            // Input lengths change on every call, so a planned memory pattern only adds memory.
            .with_memory_pattern(false)
            .map_err(model_error)?
            .commit_from_file(dir.join("model_int8.onnx"))
            .map_err(model_error)?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_error)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(model_error)?;
        // Pad to the longest input of the batch (tokenizer.json itself pads to a fixed 512).
        tokenizer.with_padding(Some(PaddingParams::default()));
        Ok(Model {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Token ids and attention mask of a batch, as (batch, length, ids, mask).
    fn encode<'s, E: Into<tokenizers::EncodeInput<'s>> + Send>(
        &self,
        inputs: Vec<E>,
    ) -> Result<(usize, usize, Vec<i64>, Vec<i64>), Error> {
        let enc = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(model_error)?;
        let (b, l) = (enc.len(), enc.first().map_or(0, |e| e.get_ids().len()));
        let ids = enc
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&v| v as i64))
            .collect();
        let mask = enc
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&v| v as i64))
            .collect();
        Ok((b, l, ids, mask))
    }

    // Both models run on one text at a time: that needs the least memory (about 1.5 GB in all instead of
    // 2.6 GB), is as fast on a CPU, and makes every score independent of the other texts (the int8 models'
    // activation scaling otherwise depends on the whole batch).

    /// Unit-length sentence embeddings.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let mut out = Vec::with_capacity(texts.len());
        let mut session = self.session.lock().unwrap();
        for text in texts {
            let (b, l, ids, mask) = self.encode(vec![*text])?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(([b, l], ids)).map_err(model_error)?,
                    "attention_mask" => Tensor::from_array(([b, l], mask)).map_err(model_error)?,
                ])
                .map_err(model_error)?;
            let (_, v) = outputs["sentence_embedding"]
                .try_extract_tensor::<f32>()
                .map_err(model_error)?;
            for row in v.chunks(v.len() / b) {
                out.push(flow::unit(row));
            }
        }
        Ok(out)
    }

    /// Reranker probability (0..1) that each sentence answers the question.
    fn rerank(&self, question: &str, sentences: &[Passage]) -> Result<Vec<f64>, Error> {
        let mut scores = Vec::with_capacity(sentences.len());
        let mut session = self.session.lock().unwrap();
        for sentence in sentences {
            let (b, l, ids, mask) = self.encode(vec![(
                question,
                &*crate::flow::scoring_text(&sentence.text),
            )])?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(([b, l], ids)).map_err(model_error)?,
                    "attention_mask" => Tensor::from_array(([b, l], mask)).map_err(model_error)?,
                    "token_type_ids" => Tensor::from_array(([b, l], vec![0i64; b * l])).map_err(model_error)?,
                ])
                .map_err(model_error)?;
            let (_, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(model_error)?;
            scores.push(flow::probability(logits[0]));
        }
        Ok(scores)
    }
}

/// Loaded models, ready to index documents and answer questions. Can be shared between threads.
pub struct QaEngine {
    embedder: Model,
    reranker: Model,
    language: LanguageCheck,
    options: QaOptions,
    /// Of the embedder's model and tokenizer files: saved documents must come from the same ones.
    embedder_fingerprint: u64,
}

impl QaEngine {
    /// Loads ONNX Runtime (once per process) and both models.
    pub fn load(models: &QaModels, options: QaOptions) -> Result<Self, Error> {
        // A second call, or an app that loaded ONNX Runtime itself, leaves the first library in place.
        ort::init_from(&models.onnxruntime)
            .map_err(model_error)?
            .commit();
        let threads = match options.threads {
            0 => std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1)),
            n => n,
        };
        Ok(QaEngine {
            embedder: Model::load(&models.embedder, threads)?,
            reranker: Model::load(&models.reranker, threads)?,
            language: LanguageCheck::new(),
            options,
            embedder_fingerprint: saved::fingerprint(&[
                models.embedder.join("model_int8.onnx"),
                models.embedder.join("tokenizer.json"),
            ])?,
        })
    }

    fn saved_key(&self) -> saved::Key {
        saved::Key {
            embedder: self.embedder_fingerprint,
            chunking: self.options.chunking,
        }
    }

    /// The prepared document as bytes, so the app can keep it (e.g. in a cache) and skip indexing it again.
    /// About 3 KB per chunk.
    pub fn save_document(&self, doc: &Document) -> Vec<u8> {
        saved::to_bytes(self.saved_key(), &doc.chunks, &doc.vectors, doc.language)
    }

    /// A document from [`Self::save_document`]. Refused ([`Error::SavedDocument`]) when the bytes are damaged,
    /// from another leafmind format, or saved with another embedder model or chunking setting: then index the
    /// document again.
    pub fn load_document(&self, bytes: &[u8]) -> Result<Document, Error> {
        let (chunks, vectors, language) = saved::from_bytes(self.saved_key(), bytes)?;
        Ok(Document::new(chunks, vectors, language))
    }

    /// Reads a PDF and prepares it for questions.
    pub fn index_pdf(&self, pdf: &[u8]) -> Result<Document, Error> {
        let pages = text::pdf_pages(pdf)?;
        let refs: Vec<(u32, &str)> = pages.iter().map(|(n, t)| (*n, t.as_str())).collect();
        self.index_pages(&refs)
    }

    /// Prepares already extracted pages (page number, markdown) for questions.
    pub fn index_pages(&self, pages: &[(u32, &str)]) -> Result<Document, Error> {
        let chunks = text::chunks(pages, self.options.chunking);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let vectors = self.embedder.embed(&texts)?;
        let language = self.language.document(&texts.join(" "));
        Ok(Document::new(chunks, vectors, language))
    }

    /// Finds the sentence of `doc` that answers `question`, or withholds the answer. About 0.1–0.25 s on a CPU.
    pub fn ask(&self, doc: &Document, question: &str) -> Result<Answer, Error> {
        if let Some(document) = doc.language
            && let Some(question) = self.language.refuse(document, question)
        {
            return Ok(Answer::WrongLanguage { document, question });
        }
        if doc.chunks.is_empty() {
            return Ok(Answer::NotFound {
                best_confidence: 0.0,
            });
        }
        let q = self.embedder.embed(&[question])?.remove(0);
        let sentences = flow::candidates(doc, question, &q);
        let scores = self.reranker.rerank(question, &sentences)?;
        Ok(flow::decide(
            &self.options,
            doc.form,
            question,
            &scores,
            &sentences,
        ))
    }
}

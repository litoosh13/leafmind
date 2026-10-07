//! The models and the question-answering flow: gte-multilingual-base embeds chunks and questions,
//! gte-multilingual-reranker-base scores (question, sentence) pairs. Both run on ONNX Runtime, which the app
//! ships and names by path.

use crate::accurate::Accurate;
use crate::language::{Language, LanguageCheck};
use crate::saved;
use crate::search::{self, Bm25};
use crate::text::{self, Chunking};
use crate::{Error, Passage};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

/// How many best chunks are split into sentences for the reranker.
const TOP_CHUNKS: usize = 8;
/// In accurate mode, how many of the gte reranker's best sentences the accurate reranker sees. On the benchmark
/// more candidates changed no answer, only the time; fewer (8, 6, 4) lost answers in real documents.
const ACCURATE_SHORTLIST: usize = 12;
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
    /// Optional, for [`QaEngine::ask_accurate`]: leafmind's export of Qwen3-Reranker-0.6B (`model.onnx`,
    /// `model.onnx.data`, `tokenizer.json`).
    pub accurate_reranker: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct QaOptions {
    /// CPU threads per model; 0 = half the logical CPUs (about the physical cores).
    pub threads: usize,
    /// Below this reranker score the answer is withheld ([`Answer::NotFound`]); the best cutoff for both
    /// modes on the benchmark.
    pub min_confidence: f64,
    /// How the document is cut into search chunks.
    pub chunking: Chunking,
}

impl Default for QaOptions {
    fn default() -> Self {
        QaOptions {
            threads: 0,
            min_confidence: 0.5,
            chunking: Chunking::Paragraph,
        }
    }
}

/// A document ready for questions. Plain data, so an app can keep it while the document is open.
#[derive(Clone, Debug)]
pub struct Document {
    chunks: Vec<Passage>,
    bm25: Bm25,
    vectors: Vec<Vec<f32>>,
    language: Option<Language>,
}

impl Document {
    /// The search chunks, in document order.
    pub fn chunks(&self) -> &[Passage] {
        &self.chunks
    }

    /// The document's language, if it could be told (questions must be asked in it).
    pub fn language(&self) -> Option<Language> {
        self.language
    }
}

/// The result of [`QaEngine::ask`].
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// The sentence that answers the question (then amendments that change it), exactly as in the document.
    Found {
        sentences: Vec<Passage>,
        confidence: f64,
    },
    /// No sentence scored high enough: the answer is withheld rather than guessed.
    NotFound { best_confidence: f64 },
    /// The question is clearly in another language than the document: ask again in the document's language.
    WrongLanguage {
        document: Language,
        question: Language,
    },
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
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                out.push(row.iter().map(|x| x / norm).collect());
            }
        }
        Ok(out)
    }

    /// Reranker probability (0..1) that each sentence answers the question.
    fn rerank(&self, question: &str, sentences: &[Passage]) -> Result<Vec<f64>, Error> {
        let mut scores = Vec::with_capacity(sentences.len());
        let mut session = self.session.lock().unwrap();
        for sentence in sentences {
            let (b, l, ids, mask) = self.encode(vec![(question, sentence.text.as_str())])?;
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
            scores.push(1.0 / (1.0 + (-logits[0] as f64).exp()));
        }
        Ok(scores)
    }
}

/// Loaded models, ready to index documents and answer questions. Can be shared between threads.
pub struct QaEngine {
    embedder: Model,
    reranker: Model,
    accurate: Option<Accurate>,
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
            accurate: match &models.accurate_reranker {
                Some(dir) => Some(Accurate::load(dir, threads)?),
                None => None,
            },
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
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        Ok(Document {
            bm25: Bm25::new(&texts),
            vectors,
            chunks,
            language,
        })
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
        Ok(Document {
            bm25: Bm25::new(&texts),
            vectors,
            chunks,
            language,
        })
    }

    /// Finds the sentence of `doc` that answers `question`, or withholds the answer. Fast: about 0.1–0.25 s on
    /// a CPU.
    pub fn ask(&self, doc: &Document, question: &str) -> Result<Answer, Error> {
        self.answer(doc, question, false)
    }

    /// Like [`Self::ask`], but a second, larger reranker (Qwen3-Reranker-0.6B, `QaModels::accurate_reranker`)
    /// picks among the gte reranker's 12 best sentences. It finds clearly more answers in real documents and
    /// paraphrased questions, at about 1 s (Apple M4) to 4 s (x86 office PC) more per question. It sees each sentence with the heading of its section, which helps it tell apart
    /// look-alike rules of different sections. Error if the model was not loaded.
    pub fn ask_accurate(&self, doc: &Document, question: &str) -> Result<Answer, Error> {
        if self.accurate.is_none() {
            return Err(Error::Model(
                "accurate mode needs QaModels::accurate_reranker".into(),
            ));
        }
        self.answer(doc, question, true)
    }

    fn answer(&self, doc: &Document, question: &str, accurate: bool) -> Result<Answer, Error> {
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
        let similarity: Vec<f64> = doc
            .vectors
            .iter()
            .map(|v| v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() as f64)
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
        let mut sentences = text::sentences(&top);
        let mut scores = self.reranker.rerank(question, &sentences)?;
        if let (true, Some(accurate)) = (accurate, &self.accurate) {
            // The accurate reranker judges the gte reranker's shortlist, kept in document order.
            let mut shortlist = search::order(&scores);
            shortlist.truncate(ACCURATE_SHORTLIST);
            shortlist.sort_unstable();
            sentences = shortlist.iter().map(|&i| sentences[i].clone()).collect();
            scores = accurate.score(question, &sentences)?;
        }
        let picked = search::pick(&scores, &sentences);
        let confidence = picked.first().map_or(0.0, |&i| scores[i]);
        Ok(if confidence >= self.options.min_confidence {
            Answer::Found {
                sentences: picked.iter().map(|&i| sentences[i].clone()).collect(),
                confidence,
            }
        } else {
            Answer::NotFound {
                best_confidence: confidence,
            }
        })
    }
}

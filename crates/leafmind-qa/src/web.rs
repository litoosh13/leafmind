//! The browser build (feature `wasm`): leafmind-qa for a web page. The page runs the two models (for example with
//! ONNX Runtime Web on WebGPU) and hands them in as JavaScript functions; reading the PDF, tokenizing, search, the
//! language check and the answer decision are the same Rust as in the native engine (`flow`).
//!
//! The two functions take an `Array` of `Int32Array`s (token ids, one input each, at most 512 tokens, already with
//! the special tokens) and resolve to a `Float32Array`:
//! - `embed`: the embedder's `sentence_embedding` rows, one after the other (inputs × size);
//! - `rerank`: the reranker's `logits`, one per input (each input is a question–sentence pair).
//!
//! How they batch, pad and where they run is up to the page.

use crate::flow::{self, Answer, Document, QaOptions};
use crate::language::{Language, LanguageCheck};
use crate::{Passage, saved, text};
use js_sys::{Array, Float32Array, Function, Int32Array, Object, Promise, Reflect};
use std::rc::Rc;
use tokenizers::{Tokenizer, TruncationParams};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};

/// Longest input the models see, in tokens (as natively).
const MAX_TOKENS: usize = 512;

struct Inner {
    embed_tokenizer: Tokenizer,
    rerank_tokenizer: Tokenizer,
    embed: Function,
    rerank: Function,
    language: LanguageCheck,
    options: QaOptions,
    /// Of the embedder id and its tokenizer: saved documents must come from the same ones.
    embedder: u64,
}

/// leafmind-qa in the browser: index documents and answer questions, with the models run by the page.
#[wasm_bindgen]
pub struct QaWeb(Rc<Inner>);

/// A document ready for questions (from `indexPdf`, `indexPages` or `loadDocument`).
#[wasm_bindgen]
pub struct WebDocument(Rc<Document>);

#[wasm_bindgen]
impl WebDocument {
    /// "english", "german", "persian" or "arabic"; undefined when it could not be told.
    pub fn language(&self) -> Option<String> {
        self.0.language().map(|l| language_name(l).into())
    }

    /// How many search chunks the document has.
    #[wasm_bindgen(js_name = chunkCount)]
    pub fn chunk_count(&self) -> usize {
        self.0.chunks().len()
    }
}

#[wasm_bindgen]
impl QaWeb {
    /// `embedTokenizer` and `rerankTokenizer`: the models' `tokenizer.json` contents. `embedderId`: names the
    /// embedder model (e.g. its file name and version), so documents saved with another one are refused.
    /// `minConfidence` / `minGap`: as `QaOptions` natively (default 0.5 / 0.15).
    #[wasm_bindgen(constructor)]
    pub fn new(
        embed_tokenizer: &str,
        rerank_tokenizer: &str,
        embedder_id: &str,
        embed: Function,
        rerank: Function,
        min_confidence: Option<f64>,
        min_gap: Option<f64>,
    ) -> Result<QaWeb, JsError> {
        let defaults = QaOptions::default();
        Ok(QaWeb(Rc::new(Inner {
            embed_tokenizer: tokenizer(embed_tokenizer)?,
            rerank_tokenizer: tokenizer(rerank_tokenizer)?,
            embed,
            rerank,
            language: LanguageCheck::new(),
            options: QaOptions {
                min_confidence: min_confidence.unwrap_or(defaults.min_confidence),
                min_gap: min_gap.unwrap_or(defaults.min_gap),
                ..defaults
            },
            embedder: saved::fingerprint_bytes(&[
                embedder_id.as_bytes(),
                embed_tokenizer.as_bytes(),
            ]),
        })))
    }

    /// Reads a PDF and prepares it for questions; resolves to a `WebDocument`.
    #[wasm_bindgen(js_name = indexPdf)]
    pub fn index_pdf(&self, pdf: Vec<u8>) -> Promise {
        let qa = self.0.clone();
        future_to_promise(async move {
            let pages = text::pdf_pages(&pdf).map_err(js_error)?;
            qa.index(pages).await
        })
    }

    /// Prepares already extracted pages (e.g. OCR text): page numbers and their texts, same order.
    #[wasm_bindgen(js_name = indexPages)]
    pub fn index_pages(&self, pages: Vec<u32>, texts: Vec<String>) -> Promise {
        let qa = self.0.clone();
        future_to_promise(async move { qa.index(pages.into_iter().zip(texts).collect()).await })
    }

    /// Answers `question` about `doc`; resolves to `{kind: "found", sentences: [{page, text, section}],
    /// confidence}`, `{kind: "not_found", bestConfidence}` or `{kind: "wrong_language", document, question}`.
    pub fn ask(&self, doc: &WebDocument, question: String) -> Promise {
        let (qa, doc) = (self.0.clone(), doc.0.clone());
        future_to_promise(async move { Ok(answer_object(&qa.ask(&doc, &question).await?)) })
    }

    /// The prepared document as bytes (same format as natively), to keep it and skip indexing next time.
    #[wasm_bindgen(js_name = saveDocument)]
    pub fn save_document(&self, doc: &WebDocument) -> Vec<u8> {
        saved::to_bytes(self.0.key(), &doc.0.chunks, &doc.0.vectors, doc.0.language)
    }

    /// A document from `saveDocument`; an error when the bytes are damaged or were saved with another embedder
    /// or setting (then index the document again).
    #[wasm_bindgen(js_name = loadDocument)]
    pub fn load_document(&self, bytes: &[u8]) -> Result<WebDocument, JsError> {
        let (chunks, vectors, language) =
            saved::from_bytes(self.0.key(), bytes).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WebDocument(Rc::new(Document::new(
            chunks, vectors, language,
        ))))
    }
}

impl Inner {
    fn key(&self) -> saved::Key {
        saved::Key {
            embedder: self.embedder,
            chunking: self.options.chunking,
        }
    }

    async fn index(&self, pages: Vec<(u32, String)>) -> Result<JsValue, JsValue> {
        let refs: Vec<(u32, &str)> = pages.iter().map(|(n, t)| (*n, t.as_str())).collect();
        let chunks = text::chunks(&refs, self.options.chunking);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let vectors = self.embed(&texts).await?;
        let language = self.language.document(&texts.join(" "));
        Ok(WebDocument(Rc::new(Document::new(chunks, vectors, language))).into())
    }

    /// As `QaEngine::ask`.
    async fn ask(&self, doc: &Document, question: &str) -> Result<Answer, JsValue> {
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
        let q = self.embed(&[question]).await?.remove(0);
        let sentences = flow::candidates(doc, question, &q);
        let scores = self.rerank(question, &sentences).await?;
        Ok(flow::decide(
            &self.options,
            doc.form,
            question,
            &scores,
            &sentences,
        ))
    }

    /// Unit-length embeddings of `texts`.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, JsValue> {
        let ids = texts
            .iter()
            .map(|t| encode(&self.embed_tokenizer, *t))
            .collect::<Result<Vec<_>, _>>()?;
        let out = call(&self.embed, &ids).await?;
        if ids.is_empty() {
            return Ok(vec![]);
        }
        if out.is_empty() || !out.len().is_multiple_of(ids.len()) {
            return Err(model_error(format!(
                "embed returned {} numbers for {} inputs",
                out.len(),
                ids.len()
            )));
        }
        Ok(out.chunks(out.len() / ids.len()).map(flow::unit).collect())
    }

    /// Reranker probability (0..1) that each sentence answers the question.
    async fn rerank(&self, question: &str, sentences: &[Passage]) -> Result<Vec<f64>, JsValue> {
        let ids = sentences
            .iter()
            .map(|s| encode(&self.rerank_tokenizer, (question, s.text.as_str())))
            .collect::<Result<Vec<_>, _>>()?;
        let logits = call(&self.rerank, &ids).await?;
        if logits.len() != ids.len() {
            return Err(model_error(format!(
                "rerank returned {} numbers for {} inputs",
                logits.len(),
                ids.len()
            )));
        }
        Ok(logits.into_iter().map(flow::probability).collect())
    }
}

/// A tokenizer as the native engine sets it up: at most 512 tokens, no padding (tokenizer.json pads to 512).
fn tokenizer(json: &str) -> Result<Tokenizer, JsError> {
    let mut t = Tokenizer::from_bytes(json.as_bytes()).map_err(|e| JsError::new(&e.to_string()))?;
    t.with_truncation(Some(TruncationParams {
        max_length: MAX_TOKENS,
        ..Default::default()
    }))
    .map_err(|e| JsError::new(&e.to_string()))?;
    t.with_padding(None);
    Ok(t)
}

fn encode<'s, E: Into<tokenizers::EncodeInput<'s>>>(
    t: &Tokenizer,
    input: E,
) -> Result<Vec<i32>, JsValue> {
    let enc = t
        .encode(input, true)
        .map_err(|e| model_error(e.to_string()))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

/// Calls a page function with the inputs and waits for its `Float32Array`; no call for no inputs.
async fn call(f: &Function, inputs: &[Vec<i32>]) -> Result<Vec<f32>, JsValue> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let batch: Array = inputs
        .iter()
        .map(|ids| Int32Array::from(&ids[..]))
        .collect();
    let result = JsFuture::from(Promise::resolve(&f.call1(&JsValue::NULL, &batch)?)).await?;
    Ok(Float32Array::new(&result).to_vec())
}

fn language_name(l: Language) -> &'static str {
    match l {
        Language::English => "english",
        Language::German => "german",
        Language::Persian => "persian",
        Language::Arabic => "arabic",
    }
}

fn answer_object(answer: &Answer) -> JsValue {
    let o = Object::new();
    let set = |k: &str, v: JsValue| {
        Reflect::set(&o, &k.into(), &v).unwrap();
    };
    match answer {
        Answer::Found {
            sentences,
            confidence,
        } => {
            set("kind", "found".into());
            set("confidence", (*confidence).into());
            let list: Array = sentences
                .iter()
                .map(|s| {
                    let p = Object::new();
                    Reflect::set(&p, &"page".into(), &s.page.into()).unwrap();
                    Reflect::set(&p, &"text".into(), &s.text.as_str().into()).unwrap();
                    let section = s
                        .section
                        .as_deref()
                        .map_or(JsValue::UNDEFINED, JsValue::from);
                    Reflect::set(&p, &"section".into(), &section).unwrap();
                    JsValue::from(p)
                })
                .collect();
            set("sentences", list.into());
        }
        Answer::NotFound { best_confidence } => {
            set("kind", "not_found".into());
            set("bestConfidence", (*best_confidence).into());
        }
        Answer::WrongLanguage { document, question } => {
            set("kind", "wrong_language".into());
            set("document", language_name(*document).into());
            set("question", language_name(*question).into());
        }
    }
    o.into()
}

fn model_error(why: String) -> JsValue {
    JsError::new(&format!("model error: {why}")).into()
}

fn js_error(e: crate::Error) -> JsValue {
    JsError::new(&e.to_string()).into()
}

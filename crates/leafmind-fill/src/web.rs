//! The browser build (feature `wasm`): leafmind-fill for a web page. The page runs the two models and hands them in
//! as JavaScript functions; the field ranking and the proposal are the same Rust as natively (`flow`).
//!
//! - `embed`: as leafmind-qa's — an `Array` of `Int32Array`s (gte-multilingual-base token ids, with the special
//!   tokens) to a `Float32Array` of the `sentence_embedding` rows, one after the other.
//! - `needle(system, tools, command)`: Needle's JSON answer (a string) from its engine (`needle_reset`,
//!   `needle_init(system, tools)`, `needle_complete(command)`), with `fill.cact` loaded.
//!
//! The functions may return promises. js/leafmind-fill-worker.js is a ready page side.

use crate::flow::{self, Form, Next, Plan};
use crate::{Change, Error, FieldKind, FormField, PageWords, Proposal, Word};
use js_sys::{Array, Float32Array, Function, Int32Array, JSON, Promise};
use serde_json::{Value, json};
use std::rc::Rc;
use tokenizers::{Tokenizer, TruncationParams};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};

struct Inner {
    tokenizer: Tokenizer,
    embed: Function,
    needle: Function,
}

/// leafmind-fill in the browser: prepare forms and propose changes, with the models run by the page.
#[wasm_bindgen]
pub struct FillWeb(Rc<Inner>);

/// A form ready for commands (from `prepare`).
#[wasm_bindgen]
pub struct WebForm(Rc<Form>);

#[wasm_bindgen]
impl FillWeb {
    /// `embedTokenizer`: gte-multilingual-base's `tokenizer.json` contents; `embed` and `needle` as above.
    #[wasm_bindgen(constructor)]
    pub fn new(
        embed_tokenizer: &str,
        embed: Function,
        needle: Function,
    ) -> Result<FillWeb, JsError> {
        let mut tokenizer = Tokenizer::from_bytes(embed_tokenizer.as_bytes())
            .map_err(|e| JsError::new(&e.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|e| JsError::new(&e.to_string()))?;
        tokenizer.with_padding(None);
        Ok(FillWeb(Rc::new(Inner {
            tokenizer,
            embed,
            needle,
        })))
    }

    /// `fields`: `[{id, page, bounds: [left, top, right, bottom], kind: "text" | "checkbox" | "choice", options?,
    /// tooltip?, readOnly?}]`; `pages`: `[{page, words: [{text, bounds}]}]` (boxes y downward, one unit). Resolves to
    /// a `WebForm`; once per document.
    pub fn prepare(&self, fields: JsValue, pages: JsValue) -> Promise {
        let fill = self.0.clone();
        future_to_promise(async move {
            let (fields, pages) = (read_fields(&fields)?, read_pages(&pages)?);
            let (prepared, texts) = flow::layout(&fields, &pages);
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let vectors = fill.embed(&refs).await?;
            Ok(WebForm(Rc::new(Form::new(prepared, vectors))).into())
        })
    }

    /// What `command` (English) would change; `today` as the form should get it. Resolves to
    /// `{kind: "actions", actions: [{field, change, alternatives}]}` (change: `{kind: "fill", value}`,
    /// `{kind: "check", checked}`, `{kind: "choose", option}`, `{kind: "clear"}` or `{kind: "copy_from", field}`),
    /// `{kind: "no_value", candidates}`, `{kind: "no_such_field"}` or `{kind: "not_supported", reason}`.
    pub fn propose(&self, form: &WebForm, command: String, today: String) -> Promise {
        let (fill, form) = (self.0.clone(), form.0.clone());
        future_to_promise(async move {
            let p = fill.propose(&form, &command, &today).await?;
            JSON::parse(&proposal_json(&p).to_string())
        })
    }
}

impl Inner {
    async fn propose(&self, form: &Form, command: &str, today: &str) -> Result<Proposal, JsValue> {
        Ok(match flow::plan(command) {
            Plan::Done(p) => p,
            Plan::SameAs { target, source } => {
                let v = self.embed(&[target, source]).await?;
                flow::copy_from(form, flow::rank(form, &v[0]), flow::rank(form, &v[1]))
            }
            Plan::Rank => {
                let q = self.embed(&[command]).await?.remove(0);
                match flow::after_rank(form, command, today, flow::rank(form, &q)) {
                    Next::Done(p) => p,
                    Next::Ask(ask) => {
                        let system = flow::system((js_sys::Date::now() / 1000.0) as i64);
                        let answer = self.needle.call3(
                            &JsValue::NULL,
                            &system.into(),
                            &ask.tools.as_str().into(),
                            &command.into(),
                        )?;
                        let answer = JsFuture::from(Promise::resolve(&answer)).await?;
                        let answer = answer
                            .as_string()
                            .ok_or_else(|| model_error("needle did not return a string".into()))?;
                        flow::finish(form, command, today, &ask, &answer)
                    }
                }
            }
        })
    }

    /// Unit-length embeddings of `texts`.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, JsValue> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let batch: Array = texts
            .iter()
            .map(|t| {
                let enc = self
                    .tokenizer
                    .encode(*t, true)
                    .map_err(|e| model_error(e.to_string()))?;
                let ids: Vec<i32> = enc.get_ids().iter().map(|&id| id as i32).collect();
                Ok(Int32Array::from(&ids[..]))
            })
            .collect::<Result<_, JsValue>>()?;
        let out =
            JsFuture::from(Promise::resolve(&self.embed.call1(&JsValue::NULL, &batch)?)).await?;
        let out = Float32Array::new(&out).to_vec();
        if out.is_empty() || !out.len().is_multiple_of(texts.len()) {
            return Err(model_error(format!(
                "embed returned {} numbers for {} inputs",
                out.len(),
                texts.len()
            )));
        }
        Ok(out
            .chunks(out.len() / texts.len())
            .map(|v| {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter().map(|x| x / norm).collect()
            })
            .collect())
    }
}

/// A JavaScript value as JSON.
fn json_of(v: &JsValue) -> Result<Value, JsValue> {
    let s = JSON::stringify(v)?.as_string().unwrap_or_default();
    serde_json::from_str(&s).map_err(|e| JsError::new(&e.to_string()).into())
}

fn bounds(v: &Value) -> Result<[f32; 4], JsValue> {
    let b: Vec<f32> = v
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default();
    b.try_into()
        .map_err(|_| JsError::new("bounds must be [left, top, right, bottom]").into())
}

fn read_fields(v: &JsValue) -> Result<Vec<FormField>, JsValue> {
    let bad = |what: &str| JsValue::from(JsError::new(&format!("field: {what}")));
    json_of(v)?
        .as_array()
        .ok_or_else(|| bad("expected an array"))?
        .iter()
        .map(|f| {
            Ok(FormField {
                id: f["id"].as_u64().ok_or_else(|| bad("id"))? as u32,
                page: f["page"].as_u64().ok_or_else(|| bad("page"))? as u32,
                bounds: bounds(&f["bounds"])?,
                kind: match f["kind"].as_str() {
                    Some("text") => FieldKind::Text,
                    Some("checkbox") => FieldKind::Checkbox,
                    Some("choice") => FieldKind::Choice {
                        options: f["options"]
                            .as_array()
                            .map(|o| {
                                o.iter()
                                    .filter_map(|x| x.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    },
                    _ => return Err(bad("kind must be \"text\", \"checkbox\" or \"choice\"")),
                },
                tooltip: f["tooltip"].as_str().map(String::from),
                read_only: f["readOnly"].as_bool().unwrap_or(false),
            })
        })
        .collect()
}

fn read_pages(v: &JsValue) -> Result<Vec<PageWords>, JsValue> {
    let bad = |what: &str| JsValue::from(JsError::new(&format!("page: {what}")));
    json_of(v)?
        .as_array()
        .ok_or_else(|| bad("expected an array"))?
        .iter()
        .map(|p| {
            Ok(PageWords {
                page: p["page"].as_u64().ok_or_else(|| bad("page"))? as u32,
                words: p["words"]
                    .as_array()
                    .ok_or_else(|| bad("words"))?
                    .iter()
                    .map(|w| {
                        Ok(Word {
                            text: w["text"].as_str().ok_or_else(|| bad("word text"))?.into(),
                            bounds: bounds(&w["bounds"])?,
                        })
                    })
                    .collect::<Result<_, JsValue>>()?,
            })
        })
        .collect()
}

fn proposal_json(p: &Proposal) -> Value {
    match p {
        Proposal::Actions(actions) => json!({
            "kind": "actions",
            "actions": actions.iter().map(|a| json!({
                "field": a.field,
                "change": match &a.change {
                    Change::Fill(v) => json!({"kind": "fill", "value": v}),
                    Change::Check(c) => json!({"kind": "check", "checked": c}),
                    Change::Choose(o) => json!({"kind": "choose", "option": o}),
                    Change::Clear => json!({"kind": "clear"}),
                    Change::CopyFrom(f) => json!({"kind": "copy_from", "field": f}),
                },
                "alternatives": a.alternatives,
            })).collect::<Vec<_>>(),
        }),
        Proposal::NoValue { candidates } => json!({"kind": "no_value", "candidates": candidates}),
        Proposal::NoSuchField => json!({"kind": "no_such_field"}),
        Proposal::NotSupported { reason } => json!({"kind": "not_supported", "reason": reason}),
    }
}

fn model_error(why: String) -> JsValue {
    JsError::new(&Error::Model(why).to_string()).into()
}

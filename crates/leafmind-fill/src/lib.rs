//! Turns an English sentence about a form ("my phone number is 0151 2345678", "tick I agree") into proposed
//! changes to the form's fields. It never writes into a document: the app shows the proposals, the user confirms,
//! the app writes. Proposals can be wrong (on unseen public forms the right field came first for about two in
//! three commands, and was among the proposal and its two alternatives for about three in four), so the app must
//! always show them before writing.
//!
//! How it decides: each field's surrounding texts (from the page's words, [`leafmind_fields::field_context`], and
//! a usable tooltip) are compared with the command by gte-multilingual-base; the best fields become the tools of
//! a small fine-tuned tool-calling model (Needle, 121M parameters), which tells filling from ticking and copies
//! the value. gte's best field of that kind is proposed, the next two are the alternatives.
//!
//! ```no_run
//! use leafmind_fill::{FieldKind, FillEngine, FillModels, FormField, PageWords, Proposal};
//! let engine = FillEngine::load(&FillModels {
//!     onnxruntime: "lib/libonnxruntime.dylib".into(),
//!     embedder: "models/gte-embed".into(),
//!     needle_library: "lib/libneedle.dylib".into(),
//!     needle_model: "models/fill.cact".into(),
//! })?;
//! # let (fields, pages): (Vec<FormField>, Vec<PageWords>) = (vec![], vec![]);
//! let form = engine.prepare(&fields, &pages)?; // once per document
//! match engine.propose(&form, "my phone number is 0151 2345678", "10/27/2026")? {
//!     Proposal::Actions(actions) => println!("{actions:?}"),
//!     other => println!("{other:?}"),
//! }
//! # Ok::<(), leafmind_fill::Error>(())
//! ```

use std::fmt;

mod decide;
#[cfg(feature = "native")]
mod engine;
mod flow;
#[cfg(all(test, feature = "native"))]
mod model_tests;
#[cfg(feature = "wasm")]
mod web;

#[cfg(feature = "native")]
pub use engine::{FillEngine, FillModels};
pub use flow::Form;
pub use leafmind_fields::Word;
#[cfg(feature = "wasm")]
pub use web::{FillWeb, WebForm};

/// What kind of field it is. Signatures and push buttons are not passed in.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldKind {
    Text,
    Checkbox,
    /// A radio group, dropdown or list, with its options as the user would name them.
    Choice {
        options: Vec<String>,
    },
}

/// A form field as the app knows it.
#[derive(Clone, Debug, PartialEq)]
pub struct FormField {
    /// The app's id for it; proposals name fields by this id.
    pub id: u32,
    pub page: u32,
    /// `[left, top, right, bottom]`, y downward, in the same unit as the page's words.
    pub bounds: [f32; 4],
    pub kind: FieldKind,
    /// The tooltip (PDF /TU), if any.
    pub tooltip: Option<String>,
    /// Never proposed (still used to tell where other fields' labels end).
    pub read_only: bool,
}

/// The words of one page with their boxes (from the PDF's text, or OCR for a scan).
#[derive(Clone, Debug, PartialEq)]
pub struct PageWords {
    pub page: u32,
    pub words: Vec<Word>,
}

/// What the command would do.
#[derive(Clone, Debug, PartialEq)]
pub enum Proposal {
    /// Changes to make after the user confirms; usually one, several when the command names several fields.
    Actions(Vec<Action>),
    /// It says which field but no value to write: ask for one. The fields that fit, best first.
    NoValue { candidates: Vec<u32> },
    /// The form has no field of the kind needed (e.g. a tick, but no checkboxes).
    NoSuchField,
    /// Something this does not do, with the reason to show.
    NotSupported { reason: &'static str },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    pub field: u32,
    pub change: Change,
    /// Up to two other fields of the same kind that might be meant, best first ("Did you mean …?").
    pub alternatives: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    Fill(String),
    Check(bool),
    Choose(String),
    Clear,
    /// Write the current value of this other field ("mailing address same as home address").
    CopyFrom(u32),
}

/// What can go wrong.
#[derive(Debug)]
pub enum Error {
    /// ONNX Runtime, the Needle library or a model could not be loaded or run (the message says why).
    Model(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Model(why) => write!(f, "model error: {why}"),
        }
    }
}

impl std::error::Error for Error {}

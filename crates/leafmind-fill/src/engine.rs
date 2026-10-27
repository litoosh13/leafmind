//! The native engine: gte-multilingual-base on ONNX Runtime (which the app ships and names by path) ranks the
//! fields, and Needle's C engine (Cactus, loaded at run time like ONNX Runtime) runs the fine-tuned `fill.cact`.

use crate::flow::{self, Form, Next, Plan};
use crate::{Error, FormField, PageWords, Proposal};
use libloading::Library;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::ffi::{CStr, CString, c_char, c_float, c_int};
use std::path::PathBuf;
use std::sync::Mutex;
use tokenizers::{Tokenizer, TruncationParams};

/// Where the app keeps the libraries and models (see THIRD_PARTY.md for the sources).
#[derive(Clone, Debug)]
pub struct FillModels {
    /// The ONNX Runtime library: `onnxruntime.dll`, `libonnxruntime.so` or `libonnxruntime.dylib` (1.30).
    pub onnxruntime: PathBuf,
    /// gte-multilingual-base, the same folder leafmind-qa uses (`model_int8.onnx`, `tokenizer.json`).
    pub embedder: PathBuf,
    /// Needle's C engine: `libneedle3.dylib`, `libneedle3.so` or `libneedle3.dll` (Cactus 3.2; scripts/fetch-qa-models.sh).
    pub needle_library: PathBuf,
    /// The fine-tuned model `fill.cact`.
    pub needle_model: PathBuf,
}

fn model_error(e: impl std::fmt::Display) -> Error {
    Error::Model(e.to_string())
}

/// gte-multilingual-base: unit-length sentence embeddings, one text at a time (as leafmind-qa runs it).
struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    fn load(dir: &std::path::Path) -> Result<Self, Error> {
        let threads = std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1));
        let session = Session::builder()
            .map_err(model_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_error)?
            .with_intra_threads(threads)
            .map_err(model_error)?
            .with_memory_pattern(false)
            .map_err(model_error)?
            .commit_from_file(dir.join("model_int8.onnx"))
            .map_err(model_error)?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_error)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(model_error)?;
        tokenizer.with_padding(None);
        Ok(Embedder {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let enc = self.tokenizer.encode(text, true).map_err(model_error)?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        let l = ids.len();
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(([1, l], ids)).map_err(model_error)?,
                "attention_mask" => Tensor::from_array(([1, l], vec![1i64; l])).map_err(model_error)?,
            ])
            .map_err(model_error)?;
        let (_, v) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(model_error)?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(v.iter().map(|x| x / norm).collect())
    }
}

type Load = unsafe extern "C" fn(*const u8, u64) -> c_int;
type Init = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> c_int;
type Complete =
    unsafe extern "C" fn(*const c_char, *const c_float, c_int, c_int, *mut c_char, c_int) -> c_int;
type Reset = unsafe extern "C" fn();
type LastError = unsafe extern "C" fn() -> *const c_char;

/// Needle's engine holds one model for the whole process and is not thread-safe: every call goes through this.
static NEEDLE: Mutex<()> = Mutex::new(());

// ponytail: one process-global Needle model; a second FillEngine replaces the first's model (same file → no harm).
struct Needle {
    init: Init,
    complete: Complete,
    reset: Reset,
    last_error: LastError,
    _lib: Library,
}

impl Needle {
    fn load(library: &std::path::Path, model: &std::path::Path) -> Result<Self, Error> {
        let weights = std::fs::read(model).map_err(model_error)?;
        let _guard = NEEDLE.lock().unwrap();
        // SAFETY: loading the app's Needle library; the symbols have the signatures of Cactus 3.2's C API.
        unsafe {
            let lib = Library::new(library).map_err(model_error)?;
            let load: Load = *lib.get(b"needle_load").map_err(model_error)?;
            let needle = Needle {
                init: *lib.get(b"needle_init").map_err(model_error)?,
                complete: *lib.get(b"needle_complete").map_err(model_error)?,
                reset: *lib.get(b"needle_reset").map_err(model_error)?,
                last_error: *lib.get(b"needle_last_error").map_err(model_error)?,
                _lib: lib,
            };
            if load(weights.as_ptr(), weights.len() as u64) < 0 {
                return Err(needle.error("needle_load"));
            }
            Ok(needle)
        }
    }

    fn error(&self, what: &str) -> Error {
        // SAFETY: the runtime owns the message; it is valid until the next call, which the lock keeps away.
        let why = unsafe { CStr::from_ptr((self.last_error)()) }.to_string_lossy();
        Error::Model(format!("{what}: {why}"))
    }

    /// Needle's JSON answer to `command` with these tools.
    fn call(&self, system: &str, tools: &str, command: &str) -> Result<String, Error> {
        let (system, tools, command) = (
            CString::new(system).map_err(model_error)?,
            CString::new(tools).map_err(model_error)?,
            CString::new(command).map_err(model_error)?,
        );
        let mut out = vec![0u8; 65536];
        let _guard = NEEDLE.lock().unwrap();
        // SAFETY: valid C strings and an output buffer of the stated size; calls serialised by the lock.
        unsafe {
            (self.reset)();
            if (self.init)(system.as_ptr(), tools.as_ptr(), std::ptr::null()) < 0 {
                return Err(self.error("needle_init"));
            }
            let n = out.len() as c_int;
            if (self.complete)(
                command.as_ptr(),
                std::ptr::null(),
                0,
                512,
                out.as_mut_ptr().cast(),
                n,
            ) < 0
            {
                return Err(self.error("needle_complete"));
            }
            Ok(CStr::from_ptr(out.as_ptr().cast())
                .to_string_lossy()
                .into_owned())
        }
    }
}

/// Loaded models, ready to prepare forms and propose changes. Can be shared between threads.
pub struct FillEngine {
    embedder: Embedder,
    needle: Needle,
}

impl FillEngine {
    /// Loads ONNX Runtime (once per process), gte and Needle with the fine-tuned model.
    pub fn load(models: &FillModels) -> Result<Self, Error> {
        ort::init_from(&models.onnxruntime)
            .map_err(model_error)?
            .commit();
        Ok(FillEngine {
            embedder: Embedder::load(&models.embedder)?,
            needle: Needle::load(&models.needle_library, &models.needle_model)?,
        })
    }

    /// Reads each field's surroundings from its page's words and embeds them. Once per document (about 10 ms
    /// per distinct text on a laptop); prepare again when fields are added or removed.
    pub fn prepare(&self, fields: &[FormField], pages: &[PageWords]) -> Result<Form, Error> {
        let (prepared, texts) = flow::layout(fields, pages);
        let vectors = texts
            .iter()
            .map(|t| self.embedder.embed(t))
            .collect::<Result<_, _>>()?;
        Ok(Form::new(prepared, vectors))
    }

    /// What `command` (English) would change. `today` is today's date as the form should get it ("10/27/2026").
    /// About 0.1 s on a laptop (a sentence about several fields may take one run per field).
    pub fn propose(&self, form: &Form, command: &str, today: &str) -> Result<Proposal, Error> {
        let p = self.propose_one(form, command, today)?;
        if let Some(parts) = flow::parts(command, &p) {
            let each = parts
                .iter()
                .map(|part| self.propose_one(form, part, today))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(merged) = flow::merge(form, &parts, each) {
                return Ok(merged);
            }
        }
        Ok(p)
    }

    fn propose_one(&self, form: &Form, command: &str, today: &str) -> Result<Proposal, Error> {
        let rank = |t: &str| Ok::<_, Error>(flow::rank(form, &self.embedder.embed(t)?));
        match flow::plan(command) {
            Plan::Done(p) => Ok(p),
            Plan::SameAs { target, source } => {
                Ok(flow::copy_from(form, rank(target)?, rank(source)?))
            }
            Plan::Rank => match flow::after_rank(form, command, today, rank(command)?) {
                Next::Done(p) => Ok(p),
                Next::Ask(ask) => {
                    let secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs() as i64);
                    let answer = self.needle.call(&flow::system(secs), &ask.tools, command)?;
                    Ok(flow::finish(form, command, today, &ask, &answer))
                }
            },
        }
    }
}

//! The native engine: gte-multilingual-base on ONNX Runtime (which the app ships and names by path) ranks the
//! fields, and Needle's C engine (Cactus, loaded at run time like ONNX Runtime) runs the fine-tuned `fill.cact`.

use crate::decide::{self, Candidate, Group};
use crate::{Action, Change, Error, FieldKind, FormField, PageWords, Proposal};
use leafmind_fields::{LabelFor, field_context, tooltip_label};
use libloading::Library;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::collections::HashMap;
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
    /// Needle's C engine: `needle.dll`, `libneedle.so` or `libneedle.dylib` (Cactus 3.2).
    pub needle_library: PathBuf,
    /// The fine-tuned model `fill.cact`.
    pub needle_model: PathBuf,
}

fn model_error(e: impl std::fmt::Display) -> Error {
    Error::Model(e.to_string())
}

/// Weight of a text by where it stands from a text field (or dropdown) and from a checkbox: a checkbox's label
/// is usually on its right. Fixed before the field-matching test on public forms (2026-10-23).
fn weight(group: Group, place: &str) -> f32 {
    match (group, place) {
        (Group::Tick, "right") => 1.0,
        (Group::Tick, "left") => 0.95,
        (Group::Tick, "above" | "inside") => 0.9,
        (Group::Tick, "below") => 0.85,
        (_, "below") => 0.9,
        (_, "right") => 0.85,
        _ => 1.0,
    }
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
    fn call(&self, tools: &str, command: &str) -> Result<String, Error> {
        let (tools, command) = (
            CString::new(tools).map_err(model_error)?,
            CString::new(command).map_err(model_error)?,
        );
        let system = CString::new(date_fact(std::time::SystemTime::now())).map_err(model_error)?;
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

/// The date line Needle's Python library puts in the system prompt, as in training: "date: 2026-10-27 Tue 10:00"
/// (UTC).
fn date_fact(now: std::time::SystemTime) -> String {
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let weekday = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
    format!(
        "date: {y}-{m:02}-{d:02} {weekday} {:02}:{:02}",
        rest / 3600,
        rest / 60 % 60
    )
}

/// A form prepared for commands: each field's surrounding texts, embedded once.
pub struct Form {
    fields: Vec<Prepared>,
    vectors: Vec<Vec<f32>>,
}

struct Prepared {
    id: u32,
    group: Group,
    options: Vec<String>,
    read_only: bool,
    /// (weight, text, index into `vectors`).
    texts: Vec<(f32, String, usize)>,
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
        let mut prepared: Vec<Prepared> = fields
            .iter()
            .map(|f| Prepared {
                id: f.id,
                group: match f.kind {
                    FieldKind::Text => Group::Text,
                    FieldKind::Checkbox => Group::Tick,
                    FieldKind::Choice { .. } => Group::Choice,
                },
                options: match &f.kind {
                    FieldKind::Choice { options } => options.clone(),
                    _ => vec![],
                },
                read_only: f.read_only,
                texts: vec![],
            })
            .collect();
        let mut texts: HashMap<String, usize> = HashMap::new();
        let mut vectors = vec![];
        let mut index = |t: String| -> Result<(String, usize), Error> {
            if let Some(&i) = texts.get(&t) {
                return Ok((t, i));
            }
            vectors.push(self.embedder.embed(&t)?);
            texts.insert(t.clone(), vectors.len() - 1);
            Ok((t, vectors.len() - 1))
        };
        let empty = vec![];
        let mut numbers: Vec<u32> = fields.iter().map(|f| f.page).collect();
        numbers.sort_unstable();
        numbers.dedup();
        for page in numbers {
            let on: Vec<usize> = (0..fields.len())
                .filter(|&i| fields[i].page == page)
                .collect();
            let boxes: Vec<([f32; 4], LabelFor)> = on
                .iter()
                .map(|&i| {
                    let label = match fields[i].kind {
                        FieldKind::Checkbox => LabelFor::Choice,
                        _ => LabelFor::Text,
                    };
                    (fields[i].bounds, label)
                })
                .collect();
            let words = pages
                .iter()
                .find(|p| p.page == page)
                .map_or(&empty, |p| &p.words);
            for (&i, ctx) in on.iter().zip(field_context(&boxes, words)) {
                if fields[i].read_only {
                    continue;
                }
                let g = prepared[i].group;
                for (place, t) in ctx {
                    let (t, v) = index(t)?;
                    prepared[i].texts.push((weight(g, place), t, v));
                }
                if let Some(t) = fields[i].tooltip.as_deref().and_then(tooltip_label) {
                    let (t, v) = index(t)?;
                    prepared[i].texts.push((1.0, t, v));
                }
            }
        }
        Ok(Form {
            fields: prepared,
            vectors,
        })
    }

    /// Every field that can be proposed, scored against `text` (fields without any text around are left out).
    fn rank(&self, form: &Form, text: &str) -> Result<Vec<Candidate>, Error> {
        let q = self.embedder.embed(text)?;
        let dot = |v: &[f32]| q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        Ok(form
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.read_only)
            .filter_map(|(i, f)| {
                let (score, text) = f
                    .texts
                    .iter()
                    .map(|(w, t, v)| (dot(&form.vectors[*v]) * w, t))
                    .max_by(|a, b| a.0.total_cmp(&b.0))?;
                Some(Candidate {
                    field: i,
                    group: f.group,
                    score,
                    text: text.clone(),
                    options: f.options.clone(),
                })
            })
            .collect())
    }

    /// What `command` (English) would change. `today` is today's date as the form should get it ("10/27/2026").
    /// About 0.1–0.2 s on a laptop.
    pub fn propose(&self, form: &Form, command: &str, today: &str) -> Result<Proposal, Error> {
        if decide::is_signing(command) {
            return Ok(Proposal::NotSupported {
                reason: "signatures are added with a signing tool, not typed",
            });
        }
        let id = |c: &Candidate| form.fields[c.field].id;
        let best_text = |cands: Vec<Candidate>, not: Option<usize>| {
            decide::shortlist(cands)
                .into_iter()
                .filter(|c| c.group == Group::Text && Some(c.field) != not)
                .collect::<Vec<_>>()
        };
        if let Some((target, source)) = decide::same_as(command) {
            let to = best_text(self.rank(form, target)?, None);
            let Some(t) = to.first() else {
                return Ok(Proposal::NoSuchField);
            };
            let from = best_text(self.rank(form, source)?, Some(t.field));
            let Some(s) = from.first() else {
                return Ok(Proposal::NoSuchField);
            };
            return Ok(Proposal::Actions(vec![Action {
                field: id(t),
                change: Change::CopyFrom(id(s)),
                alternatives: to[1..].iter().take(2).map(id).collect(),
            }]));
        }
        let cands = decide::shortlist(self.rank(form, command)?);
        let mapped = |p: Proposal| match p {
            Proposal::Actions(actions) => Proposal::Actions(
                actions
                    .into_iter()
                    .map(|a| Action {
                        field: form.fields[a.field as usize].id,
                        change: match a.change {
                            Change::CopyFrom(f) => Change::CopyFrom(form.fields[f as usize].id),
                            c => c,
                        },
                        alternatives: a
                            .alternatives
                            .iter()
                            .map(|&f| form.fields[f as usize].id)
                            .collect(),
                    })
                    .collect(),
            ),
            Proposal::NoValue { candidates } => Proposal::NoValue {
                candidates: candidates
                    .iter()
                    .map(|&f| form.fields[f as usize].id)
                    .collect(),
            },
            p => p,
        };
        if decide::is_clearing(command) {
            let Some(best) = cands
                .iter()
                .filter(|c| c.group != Group::Choice)
                .max_by(|a, b| a.score.total_cmp(&b.score))
            else {
                return Ok(Proposal::NoSuchField);
            };
            let same: Vec<&Candidate> = cands.iter().filter(|c| c.group == best.group).collect();
            return Ok(Proposal::Actions(vec![Action {
                field: id(best),
                change: if best.group == Group::Tick {
                    Change::Check(false)
                } else {
                    Change::Clear
                },
                alternatives: same[1..].iter().take(2).map(|c| id(c)).collect(),
            }]));
        }
        let (tools, labels) = decide::tools(&cands);
        let answer = self.needle.call(&tools, command)?;
        let calls = decide::calls(&answer, &cands, &labels);
        let held_back = decide::held_back_value(&answer);
        Ok(mapped(decide::decide(
            command,
            today,
            &cands,
            &calls,
            held_back.as_deref(),
        )))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_fact() {
        let at =
            |s: u64| super::date_fact(std::time::UNIX_EPOCH + std::time::Duration::from_secs(s));
        assert_eq!(at(0), "date: 1970-01-01 Thu 00:00");
        assert_eq!(at(1_793_095_200), "date: 2026-10-27 Tue 10:00");
        assert_eq!(at(951_782_400), "date: 2000-02-29 Tue 00:00");
    }
}

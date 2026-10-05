//! Accurate mode's answer picker: Qwen3-Reranker-0.6B, a small language model that reads a fixed prompt
//! (instruction, question, sentence) and answers "yes" or "no". The score is the probability of "yes" at the
//! last position, as in the model card. leafmind's export of the model (full precision: the 8-bit versions score
//! differently on x86 and ARM) returns only those two logits (input `input_ids` [1, n], output `yes_no` [1, 2]).
//! Slower than the gte reranker, so it only sees the gte reranker's best sentences.

use crate::engine::model_error;
use crate::{Error, Passage};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const SYSTEM: &str = "Judge whether the Document meets the requirements based on the Query and the Instruct \
provided. Note that the answer can only be \"yes\" or \"no\".";
/// The model card recommends a short English task description.
const INSTRUCTION: &str = "Given a question about a document, find the sentence that answers it";

pub(crate) struct Accurate {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Accurate {
    /// Loads `model.onnx` (+ its weights `model.onnx.data`) and `tokenizer.json` from `dir`.
    pub(crate) fn load(dir: &Path, threads: usize) -> Result<Self, Error> {
        let session = Session::builder()
            .map_err(model_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(model_error)?
            .with_intra_threads(threads)
            .map_err(model_error)?
            .with_inter_threads(1)
            .map_err(model_error)?
            .with_memory_pattern(false)
            .map_err(model_error)?
            .commit_from_file(dir.join("model.onnx"))
            .map_err(model_error)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_error)?;
        Ok(Accurate {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Probability (0..1) that each sentence answers the question.
    pub(crate) fn score(&self, question: &str, sentences: &[Passage]) -> Result<Vec<f64>, Error> {
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let mut scores = Vec::with_capacity(sentences.len());
        for sentence in sentences {
            // The section heading helps tell apart look-alike rules of different sections (the gte reranker
            // got worse with it, this model better).
            let text = match &sentence.section {
                Some(section) => format!("{section}: {}", sentence.text),
                None => sentence.text.clone(),
            };
            let prompt = format!(
                "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<Instruct>: {INSTRUCTION}\n\n\
                 <Query>: {question}\n\n<Document>: {text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids: Vec<i64> = self
                .tokenizer
                .encode(prompt, false)
                .map_err(model_error)?
                .get_ids()
                .iter()
                .map(|&i| i as i64)
                .collect();
            let input = Tensor::from_array(([1, ids.len()], ids)).map_err(model_error)?;
            let outputs = session
                .run(ort::inputs!["input_ids" => input])
                .map_err(model_error)?;
            let (_, logits) = outputs["yes_no"]
                .try_extract_tensor::<f32>()
                .map_err(model_error)?;
            let (yes, no) = (logits[0] as f64, logits[1] as f64);
            let top = yes.max(no);
            let (ey, en) = ((yes - top).exp(), (no - top).exp());
            scores.push(ey / (ey + en));
        }
        Ok(scores)
    }
}

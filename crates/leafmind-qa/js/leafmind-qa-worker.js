// leafmind-qa in the browser: the worker behind leafmind-qa-web.js. It holds the wasm engine (QaWeb), the two
// models (ONNX Runtime Web: WebGPU when the browser has it, else WebAssembly) and the documents, so the page's
// thread never runs model or PDF work. Messages: {id, op, ...} in, {id, ok, value | error} out.

import init, { QaWeb } from "./leafmind_qa.js";

let ort, qa, device;
const docs = new Map();
let nextDoc = 1;

// Batches of inputs of similar length (fewer padded tokens), at most this many inputs and tokens per run (faster
// than one input per run on WebGPU and on the CPU).
const BATCH_INPUTS = 16;
const BATCH_TOKENS = 4096;
const PAD = 1; // <pad> in the gte tokenizers

function batches(inputs) {
  const order = inputs.map((ids, i) => i).sort((a, b) => inputs[a].length - inputs[b].length);
  const out = [];
  let cur = [];
  for (const i of order) {
    const len = inputs[i].length;
    if (cur.length && (cur.length >= BATCH_INPUTS || (cur.length + 1) * len > BATCH_TOKENS)) {
      out.push(cur);
      cur = [];
    }
    cur.push(i);
  }
  if (cur.length) out.push(cur);
  return out;
}

// A page function for QaWeb: runs `session` on every input (padded batches, padding masked out) and returns the
// rows of `output` (the same number of values per input), in the order of the inputs.
function runner(session, output) {
  return async (inputs) => {
    const rows = new Array(inputs.length);
    for (const group of batches(inputs)) {
      const n = group.length, len = Math.max(...group.map((i) => inputs[i].length));
      const ids = new BigInt64Array(n * len).fill(BigInt(PAD)), mask = new BigInt64Array(n * len);
      group.forEach((i, row) => inputs[i].forEach((v, j) => { ids[row * len + j] = BigInt(v); mask[row * len + j] = 1n; }));
      const feed = { input_ids: new ort.Tensor("int64", ids, [n, len]), attention_mask: new ort.Tensor("int64", mask, [n, len]) };
      if (session.inputNames.includes("token_type_ids")) feed.token_type_ids = new ort.Tensor("int64", new BigInt64Array(n * len), [n, len]);
      const out = await session.run(feed);
      const data = await out[output].getData();
      const w = data.length / n;
      group.forEach((i, row) => { rows[i] = data.slice(row * w, (row + 1) * w); });
      for (const t of Object.values(out)) t.dispose?.();
    }
    const result = new Float32Array(rows.reduce((n, r) => n + r.length, 0));
    rows.reduce((at, r) => { result.set(r, at); return at + r.length; }, 0);
    return result;
  };
}

async function load({ wasmUrl, ortUrl, modelsUrl, device: want, minConfidence, minGap }) {
  ort = await import(ortUrl + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ortUrl;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = "error";
  await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  const gpu = want !== "wasm" && !!self.navigator?.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
  device = gpu ? "webgpu" : "wasm";
  if (want === "webgpu" && !gpu) throw new Error("WebGPU is not available in this browser");
  const opts = { executionProviders: [device] };
  const get = (name) => fetch(modelsUrl + name).then((r) => {
    if (!r.ok) throw new Error(`${modelsUrl + name}: HTTP ${r.status}`);
    return r;
  });
  // One after the other: ONNX Runtime Web refuses a second WebGPU session while one is being created.
  const embed = await ort.InferenceSession.create(modelsUrl + "gte-embed/model_q4f16.onnx", opts);
  const rerank = await ort.InferenceSession.create(modelsUrl + "gte-reranker/model_q4f16.onnx", opts);
  const [embedTok, rerankTok] = await Promise.all([
    get("gte-embed/tokenizer.json").then((r) => r.text()),
    get("gte-reranker/tokenizer.json").then((r) => r.text()),
  ]);
  qa = new QaWeb(embedTok, rerankTok, "gte-embed lean q4f16", runner(embed, "sentence_embedding"),
                 runner(rerank, "logits"), minConfidence, minGap);
  return { device };
}

const ops = {
  load,
  async indexPdf({ bytes }) { return keep(await qa.indexPdf(new Uint8Array(bytes))); },
  async indexPages({ pages, texts }) { return keep(await qa.indexPages(Uint32Array.from(pages), texts)); },
  async ask({ doc, question }) { return qa.ask(find(doc), question); },
  saveDocument({ doc }) { return qa.saveDocument(find(doc)); },
  loadDocument({ bytes }) { return keep(qa.loadDocument(new Uint8Array(bytes))); },
  closeDocument({ doc }) { find(doc).free(); docs.delete(doc); },
};

function keep(d) {
  const id = nextDoc++;
  docs.set(id, d);
  return { id, language: d.language(), chunks: d.chunkCount() };
}

function find(id) {
  const d = docs.get(id);
  if (!d) throw new Error("this document was closed");
  return d;
}

// One request at a time, in arrival order (the engine and the GPU work on one thing at a time anyway).
let queue = Promise.resolve();
onmessage = ({ data: { id, op, ...args } }) => {
  queue = queue.then(async () => {
    try {
      postMessage({ id, ok: true, value: await ops[op](args) });
    } catch (e) {
      postMessage({ id, ok: false, error: String(e?.message ?? e) });
    }
  });
};

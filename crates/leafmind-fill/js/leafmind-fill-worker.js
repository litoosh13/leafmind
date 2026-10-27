// leafmind-fill in the browser: the worker behind leafmind-fill-web.js. It holds the wasm engine (FillWeb), gte
// (ONNX Runtime Web), Needle's browser engine with fill.cact, and the prepared forms. Messages: {id, op, ...} in,
// {id, ok, value | error} out.

import init, { FillWeb } from "./leafmind_fill.js";

let ort, fill, device;
const forms = new Map();
let nextForm = 1;

const BATCH_INPUTS = 16;
const PAD = 1; // <pad> in the gte tokenizer

// gte for FillWeb: padded batches of similar length (padding masked out), the sentence_embedding rows in input order.
function embedder(session) {
  return async (inputs) => {
    const order = inputs.map((ids, i) => i).sort((a, b) => inputs[a].length - inputs[b].length);
    const rows = new Array(inputs.length);
    for (let at = 0; at < order.length; at += BATCH_INPUTS) {
      const group = order.slice(at, at + BATCH_INPUTS);
      const n = group.length, len = Math.max(...group.map((i) => inputs[i].length));
      const ids = new BigInt64Array(n * len).fill(BigInt(PAD)), mask = new BigInt64Array(n * len);
      group.forEach((i, row) => inputs[i].forEach((v, j) => { ids[row * len + j] = BigInt(v); mask[row * len + j] = 1n; }));
      const out = await session.run({
        input_ids: new ort.Tensor("int64", ids, [n, len]), attention_mask: new ort.Tensor("int64", mask, [n, len]),
      });
      const data = await out.sentence_embedding.getData();
      const w = data.length / n;
      group.forEach((i, row) => { rows[i] = data.slice(row * w, (row + 1) * w); });
      for (const t of Object.values(out)) t.dispose?.();
    }
    const result = new Float32Array(rows.reduce((n, r) => n + r.length, 0));
    rows.reduce((at, r) => { result.set(r, at); return at + r.length; }, 0);
    return result;
  };
}

const get = (url) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r;
});

// Needle's browser engine (an Emscripten script, not a module: imported from a Blob with an export added) with
// the fine-tuned weights; returns FillWeb's needle(system, tools, command). One model, one call at a time.
async function needle(needleUrl, modelUrl) {
  const [script, wasmBinary, weights] = await Promise.all([
    get(needleUrl + "needle.js").then((r) => r.text()),
    get(needleUrl + "needle.wasm").then((r) => r.arrayBuffer()),
    get(modelUrl).then((r) => r.arrayBuffer()),
  ]);
  const blob = URL.createObjectURL(new Blob([script + "\nexport default createNeedle;"], { type: "text/javascript" }));
  const createNeedle = (await import(blob)).default;
  URL.revokeObjectURL(blob);
  const M = await createNeedle({ wasmBinary });
  // Only ccall/cwrap/HEAPU8 are exported: strings are written and read by hand (HEAPU8 is re-read after growth).
  const str = (s) => { const b = new TextEncoder().encode(s + "\0"), p = M._malloc(b.length); M.HEAPU8.set(b, p); return p; };
  const read = (p) => { const h = M.HEAPU8; let e = p; while (h[e]) e++; return new TextDecoder().decode(h.subarray(p, e)); };
  const fail = (what) => { throw new Error(`${what}: ${read(M._needle_last_error())}`); };
  const w = M._malloc(weights.byteLength);
  M.HEAPU8.set(new Uint8Array(weights), w);
  if (M._needle_load(w, BigInt(weights.byteLength)) < 0) fail("needle_load");
  const cap = 65536, out = M._malloc(cap);
  return (system, tools, command) => {
    const args = [str(system), str(tools), str(command)];
    try {
      M._needle_reset();
      if (M._needle_init(args[0], args[1], 0) < 0) fail("needle_init");
      if (M._needle_complete(args[2], 0, 0, 512, out, cap) < 0) fail("needle_complete");
      return read(out);
    } finally {
      args.forEach((p) => M._free(p));
    }
  };
}

async function load({ wasmUrl, ortUrl, embedderUrl, needleUrl, modelUrl, device: want }) {
  ort = await import(ortUrl + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ortUrl;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = "error";
  await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  const gpu = want !== "wasm" && !!self.navigator?.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
  device = gpu ? "webgpu" : "wasm";
  if (want === "webgpu" && !gpu) throw new Error("WebGPU is not available in this browser");
  const [session, tokenizer, run] = await Promise.all([
    ort.InferenceSession.create(embedderUrl + "model_q4f16.onnx", { executionProviders: [device] }),
    get(embedderUrl + "tokenizer.json").then((r) => r.text()),
    needle(needleUrl, modelUrl),
  ]);
  fill = new FillWeb(tokenizer, embedder(session), run);
  return { device };
}

const ops = {
  load,
  async prepare({ fields, pages }) {
    const id = nextForm++;
    forms.set(id, await fill.prepare(fields, pages));
    return id;
  },
  propose({ form, command, today }) { return fill.propose(find(form), command, today); },
  closeForm({ form }) { find(form).free(); forms.delete(form); },
};

function find(id) {
  const f = forms.get(id);
  if (!f) throw new Error("this form was closed");
  return f;
}

// One request at a time, in arrival order.
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

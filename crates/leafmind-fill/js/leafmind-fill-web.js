// leafmind-fill in the browser. Keep this file next to leafmind-fill-worker.js and the wasm package
// (leafmind_fill.js, leafmind_fill_bg.wasm). All work runs in a worker, so the page stays responsive.
//
//   import { loadFill } from "./leafmind-fill/leafmind-fill-web.js";
//   const fill = await loadFill({
//     ortUrl: "/libs/onnxruntime-web/",     // folder with ONNX Runtime Web 1.30 (ort.webgpu.min.mjs and its .wasm files)
//     embedderUrl: "/models/qa/gte-embed/", // model_q4f16.onnx + tokenizer.json (the same as leafmind-qa's)
//     needleUrl: "/libs/needle/",           // Needle's browser engine: needle.js + needle.wasm
//     modelUrl: "/models/fill.cact",
//   });
//   const form = await fill.prepare(fields, pages);   // once per document
//   const p = await fill.propose(form, "my phone number is 0151 2345678", "10/27/2026");
//   // {kind: "actions", actions: [{field, change: {kind: "fill", value}, alternatives}]} | {kind: "no_value", ...}
//
// fields: [{id, page, bounds: [left, top, right, bottom], kind: "text" | "checkbox" | "choice", options?, tooltip?,
// readOnly?}]; pages: [{page, words: [{text, bounds}]}], boxes y downward, in one unit.
// leafmind downloads nothing by itself: the app serves the libraries and models and says where they are.

/**
 * Starts the worker and loads gte (WebGPU when the browser has it, else WebAssembly) and Needle.
 * @param {{ortUrl: string, embedderUrl: string, needleUrl: string, modelUrl: string,
 *          device?: "auto" | "webgpu" | "wasm"}} options  Folder URLs end with "/".
 */
export async function loadFill({ ortUrl, embedderUrl, needleUrl, modelUrl, device = "auto" }) {
  if (!ortUrl || !embedderUrl || !needleUrl || !modelUrl) throw new Error("loadFill needs ortUrl, embedderUrl, needleUrl and modelUrl");
  const worker = new Worker(new URL("./leafmind-fill-worker.js", import.meta.url), { type: "module" });
  const pending = new Map();
  let next = 1;
  worker.onmessage = ({ data: { id, ok, value, error } }) => {
    const p = pending.get(id);
    pending.delete(id);
    ok ? p.resolve(value) : p.reject(new Error(error));
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || "leafmind-fill worker failed"));
    pending.clear();
  };
  const call = (op, args = {}) =>
    new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, op, ...args });
    });
  const abs = (u) => new URL(u, location.href).href;
  const loaded = await call("load", {
    ortUrl: abs(ortUrl), embedderUrl: abs(embedderUrl), needleUrl: abs(needleUrl), modelUrl: abs(modelUrl), device,
    wasmUrl: new URL("./leafmind_fill_bg.wasm", import.meta.url).href,
  });
  return {
    /** "webgpu" or "wasm": where gte runs (Needle always runs on the CPU). */
    device: loaded.device,
    /** Reads each field's surroundings from the words and embeds them; resolves to a form handle. */
    async prepare(fields, pages) {
      return Object.freeze({ id: await call("prepare", { fields, pages }) });
    },
    /** What the command (English) would change; today as the form should get it. */
    propose: (form, command, today) => call("propose", { form: form.id, command, today }),
    /** Frees a form the app no longer needs. */
    closeForm: (form) => call("closeForm", { form: form.id }),
    /** Stops the worker and frees the models. */
    close() { worker.terminate(); },
  };
}

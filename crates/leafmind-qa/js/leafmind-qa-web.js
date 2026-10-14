// leafmind-qa in the browser. Keep this file next to leafmind-qa-worker.js and the wasm package
// (leafmind_qa.js, leafmind_qa_bg.wasm). All work runs in a worker, so the page stays responsive.
//
//   import { loadQa } from "./leafmind-qa/leafmind-qa-web.js";
//   const qa = await loadQa({
//     ortUrl: "/libs/onnxruntime-web/",   // folder with ONNX Runtime Web 1.30 (ort.webgpu.min.mjs and its .wasm files)
//     modelsUrl: "/models/qa/",          // gte-embed/ and gte-reranker/, each model_q4f16.onnx + tokenizer.json
//   });
//   const doc = await qa.indexPdf(pdfBytes);              // or qa.indexPages([1, 2], [text1, text2])
//   const answer = await qa.ask(doc, "When does the contract end?");
//   // {kind: "found", sentences: [{page, text, section}], confidence} | {kind: "not_found", bestConfidence}
//   // | {kind: "wrong_language", document, question}
//
// leafmind downloads nothing by itself: the app serves ONNX Runtime Web and the models and says where they are.

/**
 * Starts the worker and loads the two models (WebGPU when the browser has it, else WebAssembly on the CPU,
 * several threads when the page is cross-origin isolated).
 * @param {{ortUrl: string, modelsUrl: string, device?: "auto" | "webgpu" | "wasm",
 *          minConfidence?: number, minGap?: number}} options  URLs end with "/"; minConfidence / minGap as
 *          leafmind-qa's QaOptions (0.5 / 0.15).
 */
export async function loadQa({ ortUrl, modelsUrl, device = "auto", minConfidence, minGap }) {
  if (!ortUrl || !modelsUrl) throw new Error("loadQa needs ortUrl and modelsUrl");
  const worker = new Worker(new URL("./leafmind-qa-worker.js", import.meta.url), { type: "module" });
  const pending = new Map();
  let next = 1;
  worker.onmessage = ({ data: { id, ok, value, error } }) => {
    const p = pending.get(id);
    pending.delete(id);
    ok ? p.resolve(value) : p.reject(new Error(error));
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || "leafmind-qa worker failed"));
    pending.clear();
  };
  const call = (op, args = {}, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, op, ...args }, transfer);
    });
  const abs = (u) => new URL(u, location.href).href;
  const loaded = await call("load", {
    ortUrl: abs(ortUrl), modelsUrl: abs(modelsUrl), device, minConfidence, minGap,
    wasmUrl: new URL("./leafmind_qa_bg.wasm", import.meta.url).href,
  });
  const doc = (d) => Object.freeze({ id: d.id, language: d.language, chunks: d.chunks });
  return {
    /** "webgpu" or "wasm": where the models run. */
    device: loaded.device,
    /** Reads a PDF (Uint8Array or ArrayBuffer) and prepares it for questions. */
    async indexPdf(pdf) {
      const bytes = pdf instanceof ArrayBuffer ? pdf.slice(0) : pdf.slice().buffer;
      return doc(await call("indexPdf", { bytes }, [bytes]));
    },
    /** Prepares already extracted pages (e.g. OCR text): page numbers and their texts, same order. */
    async indexPages(pages, texts) {
      return doc(await call("indexPages", { pages: Array.from(pages), texts: Array.from(texts) }));
    },
    /** Answers a question about a document from indexPdf, indexPages or loadDocument. */
    ask: (d, question) => call("ask", { doc: d.id, question }),
    /** The prepared document as bytes (Uint8Array) to keep, e.g. in IndexedDB; loadDocument reads them back. */
    saveDocument: (d) => call("saveDocument", { doc: d.id }),
    /** A document from saveDocument; fails when the bytes are damaged or from other models (index again). */
    async loadDocument(bytes) {
      const b = bytes instanceof ArrayBuffer ? bytes.slice(0) : bytes.slice().buffer;
      return doc(await call("loadDocument", { bytes: b }, [b]));
    },
    /** Frees a document the app no longer needs. */
    closeDocument: (d) => call("closeDocument", { doc: d.id }),
    /** Stops the worker and frees the models. */
    close() { worker.terminate(); },
  };
}

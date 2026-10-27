# leafmind

Local document intelligence for PDF tools: find where a form can be filled, read scanned pages, and answer
questions about a document. Everything runs on the user's device — no cloud, no uploads.

leafmind is written in Rust. The field finder and question answering run natively (desktop apps) and in the
browser (WebAssembly; question answering on WebGPU); OCR is native only. It works on page images, text and PDF bytes
you give it, so it is not tied to any one app.
It is in early development.

## What it does

- **`leafmind-fields`** — finds the form fields on a page image: text boxes, checkboxes/radio buttons and
  signature areas, and (optionally) whether each one is already filled. Native and in the browser. On a blind
  test of public forms (clean and scanned, blank and filled) it finds 90.6 % of the fields; about 270 ms per
  filled scan natively, 340 ms in the browser.
- **`leafmind-qa`** — answers a question about a PDF by **picking the sentence that answers it**, so numbers
  and wording are the document's own; otherwise it says the answer is not in the document. English, German,
  Persian and Arabic. About 0.15 s per question natively; in the browser about 0.13 s on WebGPU (Apple M4 Max).
- **`leafmind-fill`** — turns an English sentence about a form ("my phone number is 0151 2345678", "tick I agree")
  into proposed changes to its fields, for the app to show and the user to confirm. On unseen public forms the
  right field comes first for about two in three sentences, and is among the proposal and its two alternatives for
  about three in four. About 0.1 s per sentence natively.
- **`leafmind-ocr`** — reads scanned pages with [Tesseract](https://github.com/tesseract-ocr/tesseract) 5 in
  the same four languages, detects the page's language, and returns paragraphs ready for `leafmind-qa`. On
  office-quality scans: 0.0 % wrong characters in English and German, about 4 % in Persian and Arabic.

## Using leafmind in your project

### 1. Add it

leafmind is not on crates.io; depend on a tag from [Releases](https://github.com/litoosh13/leafmind/releases)
(use the newest). Add only the parts you need:

```toml
[dependencies]
leafmind-fields = { git = "https://github.com/litoosh13/leafmind", tag = "v0.8.1" }
leafmind-qa = { git = "https://github.com/litoosh13/leafmind", tag = "v0.8.1" }
leafmind-ocr = { git = "https://github.com/litoosh13/leafmind", tag = "v0.8.1" }
leafmind-fill = { git = "https://github.com/litoosh13/leafmind", tag = "v0.8.1" }
```

Rust 1.98 or newer. leafmind downloads nothing and never goes online: your app ships the models and
libraries below and tells leafmind where they are.

### 2. Ship what each part needs

| Part | Your app ships | Where to get it |
|---|---|---|
| `leafmind-fields` | `form-field-v1-nano.onnx` (3.7 MB), optionally `form-field-v1-state.onnx` (0.4 MB) | `models/` in this repository |
| `leafmind-qa` | ONNX Runtime 1.30 library; `gte-embed/` and `gte-reranker/` (about 345 MB). In the browser: ONNX Runtime Web 1.30 and the q4f16 models (about 355 MB) | ONNX Runtime: [its releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0); models: `scripts/fetch-qa-models.sh <dir>` (browser: `scripts/fetch-qa-models.sh <dir> web`); browser package: `leafmind-qa-wasm-<tag>.zip` from Releases |
| `leafmind-fill` | ONNX Runtime 1.30 and `gte-embed/` (the same as `leafmind-qa`); Needle's engine library (Cactus 3.2, about 1 MB); `fill.cact` (63 MB). In the browser: ONNX Runtime Web 1.30, the q4f16 `gte-embed/`, Needle's browser engine (`needle.js`, `needle.wasm`) and `fill.cact` | `fill.cact` and the browser engine: `scripts/fetch-qa-models.sh <dir> fill`; gte: as for `leafmind-qa`; Needle's native engine: inside `python/cactus_needle-3.2.0-py3-none-<platform>.whl` (a zip) in [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3), published for macOS (Apple silicon and Intel), Windows and Linux (x86-64 and ARM); browser package: `leafmind-fill-wasm-<tag>.zip` from Releases |
| `leafmind-ocr` | Tesseract 5 library; language files (about 50 MB) | macOS: `tesseract-macos-<tag>.zip` from Releases (one file, Apple silicon and Intel, macOS 11+); Windows: the DLLs listed in [THIRD_PARTY.md](THIRD_PARTY.md); Linux: the `libtesseract5` package. Language files: `scripts/fetch-tessdata.sh <dir>` |

Licences of every model and library: [THIRD_PARTY.md](THIRD_PARTY.md).

### 3. Find form fields

```rust
use leafmind_fields::{FieldFinder, FieldKind};

let finder = FieldFinder::from_onnx(&std::fs::read("models/form-field-v1-nano.onnx")?)?
    .with_state(&std::fs::read("models/form-field-v1-state.onnx")?)?; // optional: filled or empty

let page = image::open("page.png")?.to_rgba8(); // a rendered PDF page or a scan
for field in finder.find(page.as_raw(), page.width(), page.height())? {
    let [left, top, right, bottom] = field.bounds; // pixels of this image
    match field.kind {
        FieldKind::Text => println!("text box at {left},{top}–{right},{bottom}, filled: {:?}", field.filled),
        FieldKind::Choice => println!("checkbox at {left},{top}"),
        FieldKind::Signature => println!("signature area at {left},{top}"),
    }
}
```

Create the finder once and reuse it. `find_with` takes `Options` (threshold, and `tiles: false` for about
twice the speed at fewer fields found). A page that is no form (fewer than 3 fields scoring 0.3 or more) gets no
fields, so plain text pages stay clean: on 1,180 text pages (reports, papers, contracts; clean and scanned) pages
with any field 141 → 39, fields found on public forms unchanged; `page_check: false` turns this off. Each field also
has its `score` (0–1) to filter on.

**Field labels:** `field_labels` names each field from the page's words and their boxes (PDF text or OCR words;
pure geometry, any language): a caption in the box, left of it, above or below it, the text right of a checkbox,
table column headers, comb fields (IBAN). `field_context` gives all those texts with their place instead of one
choice, and `tooltip_label` turns a field's tooltip (/TU) into a label when it reads like one. On public forms
nobody had tuned on, one chosen label is right for less than half of the fields; matching a request against
`field_context` with leafmind-qa's embedder found the right field first for about 3 in 4 English requests.

**In the browser:** unzip `leafmind-fields-wasm-<tag>.zip` from Releases (the package and both models) into
your site:

```js
import init, { FieldFinder } from './leafmind-fields-wasm/leafmind_fields.js';
await init();
const bytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const finder = new FieldFinder(await bytes('leafmind-fields-wasm/form-field-v1-nano.onnx'))
  .withState(await bytes('leafmind-fields-wasm/form-field-v1-state.onnx'));

const { data, width, height } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
const flat = finder.find(data, width, height); // 7 numbers per field:
for (let i = 0; i < flat.length; i += 7) {
  const [kind, score, left, top, right, bottom, filled] = flat.slice(i, i + 7);
  // kind 0 text, 1 checkbox, 2 signature; filled 1 / 0 (−1 without the state model)
}
```

### 4. Answer questions about a PDF

```rust
use leafmind_qa::{Answer, QaEngine, QaModels, QaOptions};

let qa = QaEngine::load(
    &QaModels {
        onnxruntime: "libs/libonnxruntime.dylib".into(), // onnxruntime.dll, libonnxruntime.so
        embedder: "models/qa/gte-embed".into(),
        reranker: "models/qa/gte-reranker".into(),
    },
    QaOptions::default(),
)?;

let doc = qa.index_pdf(&std::fs::read("contract.pdf")?)?; // once per document; keep it while it is open
match qa.ask(&doc, "When does the contract end?")? {
    Answer::Found { sentences, confidence } => {
        for s in sentences {
            println!("page {}: {} ({confidence:.2})", s.page, s.text);
        }
    }
    Answer::NotFound { .. } => println!("not in the document"),
    Answer::WrongLanguage { document, .. } => println!("please ask in {document:?}"),
}
```

Load the engine once at start (it takes about 0.8 GB of memory, measured on an Apple M4). `sentences` can hold
more than one: the answer, then amendments that change it.

leafmind-qa says "not in the document" rather than guess: when the best sentence scores below
`QaOptions::min_confidence` (0.5), or when another sentence scores almost as high (`min_gap`, 0.15). On a broad
public benchmark this keeps about 1 in 15 shown answers wrong instead of 1 in 4; lower `min_gap` to show more
answers and accept more wrong ones. In a form (at least 40 % of its sentences are short "label: value" rows), a
row whose label the question names is answered even when other rows score almost as high, and from a score of
0.4 when the question names the whole label; labels in capitals are scored in lower case (invented forms: 24 → 39
of 80 answered, none wrong). To lower `min_gap`:

```rust
let qa = QaEngine::load(&models, QaOptions { min_gap: 0.1, ..Default::default() })?;
```

To skip indexing when a document is opened again, keep its index as bytes (about 3 KB per chunk) and load it
next time. Loading refuses bytes that are damaged, from another leafmind format (it changes whenever leafmind
reads documents differently, so an old index is never used with new reading), or made with another embedder model
or chunking setting; then index the document again:

```rust
let bytes = qa.save_document(&doc); // store it where your app keeps its cache
let doc = match qa.load_document(&bytes) {
    Ok(doc) => doc,
    Err(_) => qa.index_pdf(&pdf)?, // stale or damaged: index again
};
```

**In the browser:** unzip `leafmind-qa-wasm-<tag>.zip` from Releases into your site, serve ONNX Runtime Web 1.30
(the `dist/` folder of the `onnxruntime-web` npm package) and the browser models (`scripts/fetch-qa-models.sh
<dir> web`). Everything runs in a worker, so the page stays responsive:

```js
import { loadQa } from "./leafmind-qa-wasm/leafmind-qa-web.js";
const qa = await loadQa({ ortUrl: "/libs/onnxruntime-web/", modelsUrl: "/models/qa/" }); // URLs end with "/"
const doc = await qa.indexPdf(pdfBytes);                    // or qa.indexPages([1, 2], [text1, text2])
const answer = await qa.ask(doc, "When does the contract end?");
// {kind: "found", sentences: [{page, text, section}], confidence} | {kind: "not_found", bestConfidence}
// | {kind: "wrong_language", document, question}
const bytes = await qa.saveDocument(doc);                   // e.g. into IndexedDB; qa.loadDocument(bytes) later
```

It uses WebGPU when the browser has it (about 0.13 s per question and 0.2 s per document page on an Apple M4 Max)
and the CPU otherwise (several seconds per question: a fallback). The browser models are 4-bit versions of the same
models: on the benchmark they show about 5 % fewer answers than the native ones. The package is about 33 MB (18 MB
compressed). For several CPU threads, serve the page cross-origin isolated (`Cross-Origin-Opener-Policy:
same-origin`, `Cross-Origin-Embedder-Policy: require-corp`). `loadQa` also takes `device` ("auto", "webgpu",
"wasm"), `minConfidence` and `minGap`.

### 5. Read scanned pages

```rust
use leafmind_ocr::{OcrEngine, OcrLanguage, OcrModels};

let ocr = OcrEngine::load(&OcrModels {
    tesseract: "libs/libtesseract.5.dylib".into(), // libtesseract.so.5, libtesseract-5.dll
    tessdata: "models/tessdata".into(),
})?;

let page = image::open("scan.png")?.to_rgba8();
let (w, h) = page.dimensions();
// Find the language on a page with plenty of text, then read every page with it.
let language = ocr.detect_language(page.as_raw(), w, h, Some(200))?.unwrap_or(OcrLanguage::English);
let text = ocr.read(page.as_raw(), w, h, &[language], Some(200))?;
println!("{} (confidence {:.2})", text.text, text.confidence);
for word in &text.words {
    let [left, top, right, bottom] = word.bounds; // pixels of this image, e.g. for a searchable text layer
    println!("{} on line {} at {left},{top}–{right},{bottom}", word.text, word.line);
}
```

Pass the scan's resolution (dpi) if you know it. Each call reads one page, so you can show progress and stop
between pages. On Linux with the distribution's Tesseract (built with OpenMP), start your app with
`OMP_THREAD_LIMIT=1`, as Tesseract advises when it runs inside a larger program. To ask questions about a scanned document, give the page texts to `leafmind-qa`:
`qa.index_pages(&[(1, page1_text), (2, page2_text)])`.

### 6. Fill a form from a sentence

```rust
use leafmind_fill::{FieldKind, FillEngine, FillModels, FormField, PageWords, Proposal, Word};

let fill = FillEngine::load(&FillModels {
    onnxruntime: "libs/libonnxruntime.dylib".into(),
    embedder: "models/qa/gte-embed".into(),
    needle_library: "libs/libneedle.dylib".into(), // libneedle.so, needle.dll
    needle_model: "models/fill.cact".into(),
})?;

// The document's fields and its pages' words, both with [left, top, right, bottom] boxes, y downward, in one unit.
let fields = vec![FormField {
    id: 7, // your id
    page: 1,
    bounds: [180.0, 98.0, 500.0, 112.0],
    kind: FieldKind::Text, // or Checkbox, or Choice { options }
    tooltip: None,         // the PDF's /TU, if any
    read_only: false,
}];
let words = vec![PageWords { page: 1, words: vec![
    Word { text: "Phone".into(), bounds: [50.0, 100.0, 80.0, 110.0] },
    Word { text: "number:".into(), bounds: [84.0, 100.0, 126.0, 110.0] },
] }];
let form = fill.prepare(&fields, &words)?; // once per document
match fill.propose(&form, "my phone number is 0151 2345678", "10/27/2026")? {
    Proposal::Actions(actions) => {
        for a in actions {
            // Show it, write it only after the user confirms; a.alternatives for "Did you mean …?".
            println!("field {}: {:?} (or {:?})", a.field, a.change, a.alternatives);
        }
    }
    Proposal::NoValue { candidates } => println!("what should go into field {candidates:?}?"),
    Proposal::NoSuchField => println!("the form has no field for that"),
    Proposal::NotSupported { reason } => println!("{reason}"),
}
```

`Change` is `Fill(text)`, `Check(bool)`, `Choose(option)`, `Clear` or `CopyFrom(field)` ("mailing address same as
home address": write that field's current value). The last argument is today's date as the form should get it, for
"the date is today". Signatures ("sign the form") are not proposed. English only for now. leafmind reads the labels
from the words around each field and from tooltips, so words from the PDF's text layer (or from OCR for a scan)
matter. The proposals can be wrong: always show them before writing. Needle's engine holds one model for the whole
process, so load one `FillEngine`.

In the browser, unpack `leafmind-fill-wasm-<tag>.zip` and serve it with the files from
`scripts/fetch-qa-models.sh <dir> web` and `<dir> fill`:

```js
import { loadFill } from "./leafmind-fill/leafmind-fill-web.js";
const fill = await loadFill({
  ortUrl: "/libs/onnxruntime-web/",     // ONNX Runtime Web 1.30 (ort.webgpu.min.mjs and its .wasm files)
  embedderUrl: "/models/qa/gte-embed/", // model_q4f16.onnx + tokenizer.json
  needleUrl: "/models/needle-web/",     // needle.js + needle.wasm
  modelUrl: "/models/fill/fill.cact",
});
const form = await fill.prepare(
  [{ id: 7, page: 1, bounds: [180, 98, 500, 112], kind: "text" }], // kind: "text" | "checkbox" | "choice" (+ options)
  [{ page: 1, words: [{ text: "Phone", bounds: [50, 100, 80, 110] }, { text: "number:", bounds: [84, 100, 126, 110] }] }],
);
const p = await fill.propose(form, "my phone number is 0151 2345678", "10/27/2026");
// {kind: "actions", actions: [{field: 7, change: {kind: "fill", value: "0151 2345678"}, alternatives: []}]}
// change kinds: fill, check, choose, clear, copy_from; or {kind: "no_value" | "no_such_field" | "not_supported"}
```

Everything runs in a worker. The answers match the native ones closely (on the public test forms, 93 of 144 right
against 95 natively), but Needle's browser engine runs on one CPU thread: about 1 s per sentence on an Apple M4
Max (0.1 s natively). Ticks ("tick I agree") and "label: value" sentences skip Needle and take about 50 ms, so the
average on the test forms was 0.8 s. If the page has a Content Security Policy, it must allow `blob:` scripts (Needle's
engine is a classic script, which the worker loads as a module from a Blob).

### Licence for your project

leafmind is AGPL-3.0: an app that includes it must be released under a compatible licence, with its source
available to its users (also when they use it over a network).

## Developing leafmind

```bash
cargo test                      # native tests, including the field model on a synthetic form
cargo build --target wasm32-unknown-unknown -p leafmind-fields --features wasm
```

The question-answering tests with the real models are skipped by default. To run them, fetch the models
(about 345 MB) and point the tests at them and at an ONNX Runtime 1.30 library:

```bash
scripts/fetch-qa-models.sh models/qa
LEAFMIND_QA_MODELS=$PWD/models/qa LEAFMIND_ORT=/path/to/libonnxruntime.dylib \
    cargo test -p leafmind-qa --release -- --ignored
```

The fill tests with the real models are skipped as well. They need the question-answering models above, Needle's
engine library and `fill.cact`:

```bash
LEAFMIND_QA_MODELS=$PWD/models/qa LEAFMIND_ORT=/path/to/libonnxruntime.dylib \
    LEAFMIND_NEEDLE=/path/to/libneedle.dylib LEAFMIND_FILL_MODEL=/path/to/fill.cact \
    cargo test -p leafmind-fill --release -- --ignored
```

The OCR tests with the real Tesseract are skipped by default as well. They need Tesseract 5 (for example
`brew install tesseract`) and the language files:

```bash
scripts/fetch-tessdata.sh models/tessdata
LEAFMIND_TESSERACT=/opt/homebrew/lib/libtesseract.5.dylib LEAFMIND_TESSDATA=$PWD/models/tessdata \
    cargo test -p leafmind-ocr -- --ignored
```

For speed in the browser, build WebAssembly with SIMD: `RUSTFLAGS="-C target-feature=+simd128"`.

## License

AGPL-3.0. See [LICENSE](LICENSE).

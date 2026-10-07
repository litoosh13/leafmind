# leafmind

Local document intelligence for PDF tools: find where a form can be filled, read scanned pages, and answer
questions about a document. Everything runs on the user's device — no cloud, no uploads.

leafmind is written in Rust. The field finder runs natively (desktop apps) and in the browser
(WebAssembly); question answering is native only (desktop). It works on page images, text and PDF bytes
you give it, so it is not tied to any one app.
It is in early development.

## What it does

- **`leafmind-fields`** — finds the form fields on a page image: text boxes, checkboxes/radio buttons and
  signature areas, and (optionally) whether each one is already filled. Native and in the browser. On a blind
  test of public forms (clean and scanned, blank and filled) it finds 90.6 % of the fields; about 270 ms per
  filled scan natively, 340 ms in the browser.
- **`leafmind-qa`** — answers a question about a PDF by **picking the sentence that answers it**, so numbers
  and wording are the document's own; otherwise it says the answer is not in the document. English, German,
  Persian and Arabic. Fast mode about 0.15 s per question; the optional accurate mode about 1 s (Apple M4)
  and finds clearly more answers in real documents.
- **`leafmind-ocr`** — reads scanned pages with [Tesseract](https://github.com/tesseract-ocr/tesseract) 5 in
  the same four languages, detects the page's language, and returns paragraphs ready for `leafmind-qa`. On
  office-quality scans: 0.0 % wrong characters in English and German, about 4 % in Persian and Arabic.

## Using leafmind in your project

### 1. Add it

leafmind is not on crates.io; depend on a tag from [Releases](https://github.com/litoosh13/leafmind/releases)
(use the newest). Add only the parts you need:

```toml
[dependencies]
leafmind-fields = { git = "https://github.com/litoosh13/leafmind", tag = "v0.3.1" }
leafmind-qa = { git = "https://github.com/litoosh13/leafmind", tag = "v0.3.1" }
leafmind-ocr = { git = "https://github.com/litoosh13/leafmind", tag = "v0.3.1" }
```

Rust 1.98 or newer. leafmind downloads nothing and never goes online: your app ships the models and
libraries below and tells leafmind where they are.

### 2. Ship what each part needs

| Part | Your app ships | Where to get it |
|---|---|---|
| `leafmind-fields` | `form-field-v1-nano.onnx` (3.7 MB), optionally `form-field-v1-state.onnx` (0.4 MB) | `models/` in this repository |
| `leafmind-qa` | ONNX Runtime 1.30 library; `gte-embed/` and `gte-reranker/` (about 680 MB); optionally `qwen3-reranker/` for accurate mode (2.4 GB) | ONNX Runtime: [its releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0); models: `scripts/fetch-qa-models.sh <dir> [accurate]` |
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
twice the speed at fewer fields found).

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
        accurate_reranker: None, // or Some("models/qa/qwen3-reranker".into()) for ask_accurate
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

Load the engine once at start (it takes about 1.5 GB of memory; 3.5 GB with accurate mode). `ask_accurate`
works the same way. `sentences` can hold more than one: the answer, then amendments that change it.

To skip indexing when a document is opened again, keep its index as bytes (about 3 KB per chunk) and load it
next time. Loading refuses bytes that are damaged, from another leafmind format, or made with another embedder
model or chunking setting; then index the document again:

```rust
let bytes = qa.save_document(&doc); // store it where your app keeps its cache
let doc = match qa.load_document(&bytes) {
    Ok(doc) => doc,
    Err(_) => qa.index_pdf(&pdf)?, // stale or damaged: index again
};
```

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

### Licence for your project

leafmind is AGPL-3.0: an app that includes it must be released under a compatible licence, with its source
available to its users (also when they use it over a network).

## Developing leafmind

```bash
cargo test                      # native tests, including the field model on a synthetic form
cargo build --target wasm32-unknown-unknown -p leafmind-fields --features wasm
```

The question-answering tests with the real models are skipped by default. To run them, fetch the models
(about 680 MB) and point the tests at them and at an ONNX Runtime 1.30 library:

```bash
scripts/fetch-qa-models.sh models/qa            # add "accurate" for accurate mode's model (2.4 GB)
LEAFMIND_QA_MODELS=$PWD/models/qa LEAFMIND_ORT=/path/to/libonnxruntime.dylib \
    cargo test -p leafmind-qa --release -- --ignored
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

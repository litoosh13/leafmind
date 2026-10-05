# leafmind

Local document intelligence for PDF tools: find where a form can be filled, read scanned pages, and answer
questions about a document. Everything runs on the user's device — no cloud, no uploads.

leafmind is written in Rust. The field finder runs natively (desktop apps) and in the browser
(WebAssembly); question answering is native only (desktop). It works on page images, text and PDF bytes
you give it, so it is not tied to any one app.
It is in early development.

## Parts

- `crates/leafmind-fields` — finds form fields (text boxes, checkboxes/radio buttons, signature areas) on a
  page image with Nutrient's `form-field-v1-nano` detector (0.9 M parameters) run by
  [tract](https://github.com/sonos/tract). About 50 ms per page natively and 100 ms in the browser
  (WebAssembly with SIMD) on a recent laptop. Its default threshold (0.2, the model card says 0.3) keeps far
  more fields on scans and filled forms: on simulated scans of filled public forms 78 % instead of 64 %; a box
  mostly covered by a stronger box of another kind is dropped (a fifth fewer wrong boxes). With
  Nutrient's optional `form-field-v1-state` model (0.4 MB) it also says whether each field is already filled
  (right for 92–100 % of fields on clean and scanned, blank and filled forms; about 40 ms more per page).

- `crates/leafmind-qa` — answers a question about a PDF by **picking the sentence that answers it** (it
  never writes an answer, so numbers and wording are the document's own), or says it is not in the
  document, or asks for the question in the document's language. Pipeline: PDF text (pdf-inspector, with a
  patch that keeps the Persian zero-width non-joiner) → paragraphs → keyword search (BM25) + embedding search
  (gte-multilingual-base), merged by rank fusion → the reranker gte-multilingual-reranker-base scores the
  sentences of the best paragraphs. Languages: English, German, Persian, Arabic (language check by lingua).
  Runs on ONNX Runtime, which the app ships. About 0.15 s per question on an Apple M4 (4 threads); the models
  take about 1.5 GB of memory while running.
  An optional **accurate mode** (`ask_accurate`) lets a second, larger reranker (Qwen3-Reranker-0.6B in
  leafmind's own full-precision export, 2.4 GB) choose among the 12 best sentences, each read together with the
  heading of its section. On the project's synthetic test set it answers 96 of 117 questions with 1 wrong
  (fast mode: 90, 1 wrong), and it finds clearly more answers in real documents: on the real-document test set
  40 of 51 with 4 wrong, against 31 with 8 wrong for the fast mode (or 23 with none wrong at a stricter cutoff).
  It takes about 1 s per question on an Apple M4 and about 4 s on an x86 office PC
  (Intel i5-12500), and about 3.5 GB of memory in all. Its scores are the same on both kinds of CPU (the 8-bit
  versions of the model were not).

- `crates/leafmind-ocr` — reads the text of scanned pages with [Tesseract](https://github.com/tesseract-ocr/tesseract)
  5 in English, German, Persian and Arabic. It detects a page's script and language (reading with the one
  right language is clearly better than reading with several), and groups the words into lines and
  paragraphs itself, so the text is ready for `leafmind-qa`. The app ships the Tesseract library and its
  language files; pages come in as images. On simulated office scans (200 dpi) it misreads 0.0 % of the
  characters in English and German and about 4 % in Persian and Arabic, at about 0.85 s per page on an
  Apple M4; questions about the scanned documents are answered almost as well as about the originals.
  Pages are read one call at a time, so an app shows progress and can stop between pages. Shipping Tesseract:
  on Windows the DLL set from UB Mannheim's build, on macOS `scripts/bundle-tesseract-macos.sh` (copies
  Homebrew's Tesseract and its libraries into one folder an app can ship), on Linux the distribution's
  package; the libraries and their licences are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Building and testing

```bash
cargo test                      # native tests, including the field model on a synthetic form
cargo build --target wasm32-unknown-unknown -p leafmind-fields --features wasm
```

The question-answering tests with the real models are skipped by default. To run them, fetch the models
(about 680 MB) and point the tests at them and at an ONNX Runtime 1.30 library:

```bash
scripts/fetch-qa-models.sh models/qa            # add "accurate" for accurate mode's model (2.4 GB)
LEAFMIND_QA_MODELS=models/qa LEAFMIND_ORT=/path/to/libonnxruntime.dylib \
    cargo test -p leafmind-qa --release -- --ignored
```

The OCR tests with the real Tesseract are skipped by default as well. They need Tesseract 5 (for example
`brew install tesseract`) and the language files:

```bash
scripts/fetch-tessdata.sh models/tessdata
LEAFMIND_TESSERACT=/opt/homebrew/lib/libtesseract.5.dylib LEAFMIND_TESSDATA=models/tessdata \
    cargo test -p leafmind-ocr -- --ignored
```

For speed in the browser, build WebAssembly with SIMD: `RUSTFLAGS="-C target-feature=+simd128"`.

## Models

See [THIRD_PARTY.md](THIRD_PARTY.md). The field detector (`models/form-field-v1-nano.onnx`) is
Apache-2.0, by Nutrient. The question-answering models (gte-multilingual-base and
gte-multilingual-reranker-base, Apache-2.0, by Alibaba) are not in the repository; `scripts/fetch-qa-models.sh`
downloads them. The Tesseract language files (Apache-2.0) are not in the repository either;
`scripts/fetch-tessdata.sh` downloads them.

## License

AGPL-3.0. See [LICENSE](LICENSE).

# Third-party models and data

## models/form-field-v1-nano.onnx

- **What:** form-field detector (YOLOX-Nano, 0.9 M parameters; classes Text, ChoiceButton, Signature),
  ONNX with box decoding built in, input 1×3×640×640.
- **Source:** https://huggingface.co/nutrientdocs/form-field-v1-nano (file `model.onnx`, downloaded
  2026-09-23), by Nutrient (https://nutrient.io/).
- **Licence:** Apache License 2.0 (model card). Trained on synthetic form renders (no personal data).
- **SHA-256:** `d7ab44396154af79409c9f8a4fded6c9aa907ae6d283fbe5d2b533d3b7314ec1`
- **Changes:** none. Licence text: [models/LICENSE-APACHE-2.0](models/LICENSE-APACHE-2.0).

## models/form-field-v1-state.onnx

- **What:** filled/empty classifier for detected form fields (depthwise-separable CNN, 96 k parameters),
  ONNX, input a 1×1×40×160 grey field crop, output two logits (empty, filled).
- **Source:** https://huggingface.co/nutrientdocs/form-field-v1-state, revision
  `62b1049a066d63eacf0f3542dc155b11c7284883` (file `model.onnx`, downloaded 2026-10-01), by Nutrient
  (https://nutrient.io/).
- **Licence:** Apache License 2.0 (model card). Trained on synthetic form renders (no personal data).
- **SHA-256:** `7afb50c85247174b5384e4fa857b8a56a1736a40057981904f995952c2444974`
- **Changes:** none. Licence text: [models/LICENSE-APACHE-2.0](models/LICENSE-APACHE-2.0).

## third_party/pdf-inspector (patched copy of pdf-inspector 1.24.0)

- **What:** Rust library that turns PDF pages into markdown text; used by `leafmind-qa` as a path
  dependency (so projects that use leafmind get the patched copy too). Includes the CMap tables in `external/bcmaps/` (data compiled into the library).
- **Source:** https://crates.io/crates/pdf-inspector/1.24.0 (upstream https://github.com/firecrawl/pdf-inspector),
  downloaded 2026-09-24. `.crate` SHA-256:
  `e22dc125a533d212c847c8c85e4fcb7358f4384869ef76b2b8721f039b1b633a`.
- **Licence:** MIT (`third_party/pdf-inspector/LICENSE`); CMap tables © Adobe Systems, BSD-style licence in
  `third_party/pdf-inspector/external/bcmaps/LICENSE`.
- **Changes:** keeps the zero-width non-joiner inside Persian/Arabic-script and Indic words; decodes form-field
  names and values as PDF text strings; sends scans with a few typed lines on top to OCR — see
  `third_party/pdf-inspector/LEAFMIND_PATCH.md` and its three `.patch` files.

## Question-answering models (not in the repository; `scripts/fetch-qa-models.sh` downloads them)

`leafmind-qa` needs two models that the app ships next to it. The script fetches them from
https://huggingface.co/litoo13/leafmind-gte-multilingual-4lang, revision `466fd26d9e751de3a57adc3ae222115865ca8cc4`,
and checks these SHA-256 sums. They are leafmind's vocabulary-pruned copies of the int8 ONNX conversions ("… with
ONNX weights to be compatible with Transformers.js") by Hugging Face's onnx-community of Alibaba's models; the
conversions state no licence of their own, so the originals' licence applies: **Apache License 2.0** (text:
[models/LICENSE-APACHE-2.0](models/LICENSE-APACHE-2.0); the originals ship no NOTICE file). Checked 2026-09-27.
**Changes (2026-10-13):** only two tables are cut, the network is unchanged: the word-embedding table keeps 72,928 of
250,002 word pieces (special tokens, pieces without letters, Arabic-script, Greek and one-character pieces, and the
Latin pieces English/German training text uses), each `tokenizer.json` keeps the same pieces with new ids, and the
rotary position tables keep the first 1,024 of 65,536 positions. The repository's model card and `build/` scripts
describe and reproduce it. On leafmind's broad benchmark (4,190 questions) the answers are as before (right 1,439 →
1,444, wrong 102 → 103), at half the memory.

### gte-embed — gte-multilingual-base (search)

- **Original:** https://huggingface.co/Alibaba-NLP/gte-multilingual-base (Apache-2.0), by Alibaba (Tongyi Lab).
- **Converted from:** https://huggingface.co/onnx-community/gte-multilingual-base, revision
  `2edbf5e672aab465f9ed4c154a8b61791c082c69` (`onnx/model_int8.onnx`
  `ab2bd164ebd8ca9003dc49a981b611e849b5d326f504c8873ba76e07fa6c0082`, `tokenizer.json`
  `3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20`).
- **Files** (folder `gte-embed/`):
  - `model_int8.onnx` (171,260,482 bytes) — `5668afdae8b748de54ef85b7a13a711005ccf2a7a08806159f78b30c5f722cbf`
  - `tokenizer.json` — `c1bafc0e9c74e6cefcd032df267ce8603c7a288a771e517d9c8823e0a7de8a37`
  - browser build (revision `a789eef86239fee01ba034413e5af09291f28506`): `model_q4f16.onnx` (176,633,173 bytes) —
    `aa9ae652c7f6effbfc7aa57465f6072256926c944922dee3845cec456168864f`, cut from onnx-community's
    `onnx/model_q4f16.onnx` (`9726ec27b72918ca0ed491f7f28bb52e6a5c7b8e47a49bd5f6c08a93675a6b55`)

### gte-reranker — gte-multilingual-reranker-base (picks the answer sentence)

- **Original:** https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base (Apache-2.0), by Alibaba.
- **Converted from:** https://huggingface.co/onnx-community/gte-multilingual-reranker-base, revision
  `ee64367e35a2db0da46bb6497e13a18f8bd585cb` (`onnx/model_int8.onnx`
  `ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35`, `tokenizer.json`
  `3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98`).
- **Files** (folder `gte-reranker/`):
  - `model_int8.onnx` (171,799,885 bytes) — `4e58a5fd59990fd78afc8a509f27d4f799d8b8723ceea00e0a07fd6f316fb542`
  - `tokenizer.json` — `e26ac9905f59291e2540981b82ca2e1022e9699b2456324f66ddba4804272847`
  - browser build (revision `a789eef86239fee01ba034413e5af09291f28506`): `model_q4f16.onnx` (177,762,498 bytes) —
    `8514babe9d028c6314a0ff8d6495bb778b55f54d7d4d3928589314a32c833188`, cut from onnx-community's
    `onnx/model_q4f16.onnx` (`d24aacb414ccd10944c00622eb2d36c7f90019a8c8bdb882c320dc20f746abec`)

### ONNX Runtime (shipped by the app)

- The models run on ONNX Runtime 1.30 (https://github.com/microsoft/onnxruntime, MIT), loaded at run time
  from a path the app gives (`QaModels::onnxruntime`); leafmind does not include it.
- In the browser (`leafmind-qa-wasm-<tag>.zip`) the models run on ONNX Runtime Web 1.30
  (https://github.com/microsoft/onnxruntime, npm `onnxruntime-web`, MIT), which the app serves and names by URL
  (`loadQa({ortUrl})`); the package does not include it. The package's `.wasm` contains leafmind-qa compiled with
  its Rust dependencies, among them pdf-inspector (MIT, with the CMap tables above), lingua (Apache-2.0), tokenizers
  (Apache-2.0) and wasm-bindgen (MIT or Apache-2.0).

## OCR: Tesseract and its language files (not in the repository; `scripts/fetch-tessdata.sh` downloads the files)

`leafmind-ocr` loads the Tesseract 5 library that the app ships (https://github.com/tesseract-ocr/tesseract,
Apache License 2.0; it depends on Leptonica, BSD-2-Clause, and image libraries under their own licences). The
language files, checked 2026-09-28, are Apache License 2.0
([models/LICENSE-APACHE-2.0](models/LICENSE-APACHE-2.0)):

- https://github.com/tesseract-ocr/tessdata_best at commit `e12c65a915945e4c28e237a9b52bc4a8f39a0cec`:
  - `eng.traineddata` — `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba`
  - `deu.traineddata` — `8407331d6aa0229dc927685c01a7938fc5a641d1a9524f74838cdac599f0d06e`
  - `fas.traineddata` — `99e420969b5ddd2cb135b416316a7ed417c59c4faf9e0d28941348f6448114df`
  - `ara.traineddata` — `ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896`
- https://github.com/tesseract-ocr/tessdata at commit `ced78752cc61322fb554c280d13360b35b8684e4`:
  - `osd.traineddata` (orientation and script detection) —
    `e19f2ae860792fdf372cf48d8ce70ae5da3c4052962fe22e9de1f680c374bb0e`

### Shipping the Tesseract library with an app

leafmind does not include Tesseract; an app that ships it also ships its dependencies, each under its own
licence. The lists below were checked on 2026-10-01 against the builds leafmind was tested with. Licences are
as the projects publish them; check each package's licence file before a release.

**Windows** — the 26 DLLs `libtesseract-5.dll` needs, from UB Mannheim's installer
`tesseract-ocr-w64-setup-5.4.0.20240606.exe` (https://github.com/UB-Mannheim/tesseract, MSYS2 builds; the
installer itself is not run, its files are unpacked). Versions read from the DLLs where they state one.

| DLL | Project (version) | Licence |
|---|---|---|
| `libtesseract-5.dll` | Tesseract (5.4.0) | Apache-2.0 |
| `libleptonica-6.dll` | Leptonica | BSD-2-Clause |
| `libarchive-13.dll` | libarchive (3.7.4) | BSD-2-Clause |
| `libb2-1.dll` | BLAKE2 reference (libb2) | CC0-1.0 |
| `libbz2-1.dll` | bzip2 (1.0.8) | bzip2-1.0.6 |
| `libcrypto-3-x64.dll` | OpenSSL (3.3.1) | Apache-2.0 |
| `libdeflate.dll` | libdeflate | MIT |
| `libexpat-1.dll` | Expat (2.6.2) | MIT |
| `libgcc_s_seh-1.dll`, `libstdc++-6.dll` | GCC runtime (14.1.0) | GPL-3.0-or-later WITH GCC-exception-3.1 |
| `libgif-7.dll` | GIFLIB | MIT |
| `libiconv-2.dll` | GNU libiconv | **LGPL-2.1-or-later** |
| `libjbig-0.dll` | JBIG-KIT (2.1) | **GPL-2.0-or-later** |
| `libjpeg-8.dll` | libjpeg-turbo (3.0.3) | IJG AND BSD-3-Clause AND Zlib |
| `libLerc.dll` | LERC | Apache-2.0 |
| `liblz4.dll` | LZ4 (1.9.4) | BSD-2-Clause |
| `liblzma-5.dll` | XZ Utils (5.6.2) | 0BSD |
| `libopenjp2-7.dll` | OpenJPEG (2.5.2) | BSD-2-Clause |
| `libpng16-16.dll` | libpng (1.6.43) | libpng-2.0 |
| `libsharpyuv-0.dll`, `libwebp-7.dll`, `libwebpmux-3.dll` | libwebp | BSD-3-Clause |
| `libtiff-6.dll` | LibTIFF (4.6.0) | libtiff |
| `libwinpthread-1.dll` | mingw-w64 winpthreads | MIT AND BSD-3-Clause |
| `libzstd.dll` | Zstandard (1.5.6) | BSD-3-Clause OR GPL-2.0-only |
| `zlib1.dll` | zlib (1.3.1) | Zlib |

What this asks of an app (not legal advice):
- Ship every licence text with the app (MSYS2 keeps them in `share/licenses/<package>/`; the source packages are
  at https://packages.msys2.org).
- **libiconv (LGPL-2.1):** keep it as the separate, replaceable DLL it is, and say where its source is.
- **JBIG-KIT (GPL-2.0-or-later)**, pulled in by LibTIFF: an app that ships it must offer its source and the
  combination falls under the GPL. leafmind itself is AGPL-3.0, which GPL-2.0-or-later code can join (through
  GPL-3.0). An app that wants to avoid it needs a LibTIFF built without JBIG support.
- The GCC runtime's exception allows shipping `libgcc_s` and `libstdc++` with any program; Zstandard can be
  used under its BSD licence.

**macOS** — each release has `tesseract-macos-<tag>.zip`, built in CI by `scripts/build-tesseract-macos.sh`
(the same script builds it locally): Tesseract 5.5.3 (https://github.com/tesseract-ocr/tesseract, source
`5.5.3.tar.gz` SHA-256 `9218e62793116d42a9f6d14cd9348518b27f382096eea3d0f2d1a24616bb5884`, Apache-2.0) with
Leptonica 1.87.0 linked in (https://github.com/DanBloomberg/leptonica, source `leptonica-1.87.0.tar.gz` SHA-256
`c73363397f96eb1295602bf44d708a994ad42046c791bf03ea0505d829bdb6a7`, BSD-2-Clause), as one universal
`libtesseract.5.dylib` (Apple silicon and Intel, macOS 11 or newer). leafmind gives Tesseract raw pixels, so the
build has no image formats, libarchive or libcurl: the library needs only macOS's own libSystem and libc++, and
these two licences (shipped beside it as `LICENSE-tesseract` and `LICENSE-leptonica`) are all. Tested
2026-10-06: the OCR tests pass on Apple silicon and as an Intel build under Rosetta, and on the 62 benchmark scans
it reads exactly the same text as Homebrew's Tesseract 5.5.3, slightly faster (1.06 against 1.16 s per page).

**Linux** — distributions package Tesseract 5 (Debian/Ubuntu `libtesseract5`, Fedora `tesseract-libs`); an app
can depend on that package and give leafmind its path (e.g. `/usr/lib/x86_64-linux-gnu/libtesseract.so.5`), or
bundle the libraries. Tested in CI on Ubuntu (`libtesseract5`).

## Fill commands: Needle (not in the repository; the app ships it for `leafmind-fill`)

- **Base:** https://huggingface.co/Cactus-Compute/needle3, revision `b274efcb` (121M parameters, Apache-2.0), by
  Cactus Compute.
- **Engine:** Cactus 3.2.0 (Apache-2.0), loaded at run time; the library is inside the platform wheels
  `python/cactus_needle-3.2.0-py3-none-<platform>.whl` of the same repository at revision
  `d6c7d6a2e9655505e0f8e23339356dd1705900eb` (`scripts/fetch-qa-models.sh <dir> needle-<platform>` downloads the
  wheel, checks its SHA-256 and unpacks the library). Tested: macOS arm64 `libneedle3.dylib` (1,070,448 bytes) — `6794ef011da58f6138a2c1e7103dcb73b401ddfb9aa67cc28b31fadded7ff780`. It
  links only the system libraries and has no network code (Cactus's Python package, not used by leafmind, sends
  usage counts unless `NEEDLE_TELEMETRY=0`).
- **Browser engine:** the same repository at revision `d6c7d6a2e9655505e0f8e23339356dd1705900eb`, `wasm/needle.js`
  (62,082 bytes) — `a675c66f506a1b6f922d852c0cc0b82a843ce9d0fab6a80a408106f516df126f` and `wasm/needle.wasm`
  (930,545 bytes) — `7b73d09c17923bf81d6bdb82e19b3c343b8994f83d512ddec52eec9f6b029802` (Emscripten build; it fetches
  nothing but its own `.wasm`, which leafmind's worker hands in). `scripts/fetch-qa-models.sh <dir> fill` downloads
  them with `fill.cact`.
- **Fine-tune:** https://huggingface.co/litoo13/leafmind-fill-needle, revision
  `c52f3d0a11f38e6a01d094090501a35c90f2882b`: leafmind's LoRA fine-tune (rank 32, 2 epochs) on 6,000 invented
  form-fill examples (made-up forms, labels and values; no real documents or personal data).
- **File:** `fill.cact` (63,437,076 bytes) — `6198a4579107fc6b0546816a547f6c8594979ba2ebe33a4a4b2c8643536c35fd`
- **Licence:** Apache License 2.0, like the base ([models/LICENSE-APACHE-2.0](models/LICENSE-APACHE-2.0)).

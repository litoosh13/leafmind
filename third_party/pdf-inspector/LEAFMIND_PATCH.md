# leafmind's patched copy of pdf-inspector 1.24.0

This folder is pdf-inspector 1.24.0 as published on crates.io
(https://crates.io/crates/pdf-inspector/1.24.0, upstream https://github.com/firecrawl/pdf-inspector;
`.crate` SHA-256 `e22dc125a533d212c847c8c85e4fcb7358f4384869ef76b2b8721f039b1b633a`), MIT (see `LICENSE`).
The CMap tables in `external/bcmaps/` are Adobe's, under the licence in `external/bcmaps/LICENSE`.
The workspace `Cargo.toml` uses it as a path dependency (not `[patch.crates-io]`, which only applies to the
top-level project), so projects that depend on leafmind get the patched copy too.

## Why

pdf-inspector removes every zero-width non-joiner (ZWNJ, U+200C) and zero-width joiner (U+200D) from
extracted text. In Persian (and other Arabic-script and Indic languages) the ZWNJ is part of the spelling:
without it words run together ("میکنند" instead of "می‌کنند"), so sentences shown to the reader are
misspelled. Some PDF producers (e.g. Chrome) also write each ZWNJ as a text run of its own.

## Changes (`leafmind-keep-zwnj.patch`)

- `src/text_utils.rs`: `expand_ligatures` keeps a joiner that sits between two letters when at least one of
  them is Arabic-script or Indic, and keeps a text run that is only joiners; new `strip_stray_joiners`
  removes joiners that end up anywhere else once the page text is assembled. Three unit tests added.
- `src/lib.rs`: page markdown from `extract_pages_markdown` goes through `strip_stray_joiners`.

Measured on invented test PDFs: Persian ZWNJ kept 138 of 142 (unpatched: 0); Arabic, German and English
output byte-identical to the unpatched crate.

## Form fields with non-ASCII names or values (`leafmind-form-field-strings.patch`)

pdf-inspector read the names (`/T`) and values (`/V`) of filled form fields as UTF-8. They are PDF text
strings — PDFDocEncoding, or UTF-16 with a byte-order mark as Acrobat writes non-ASCII values — so a field
named "Größe" or filled in with "Jürgen" came out with U+FFFD replacement characters. The page then counted
as garbled text (`needs_ocr`, `suspected_garbled_text`) and **its whole markdown was empty**, printed text
included.

- `src/extractor/links.rs`: field names and text/choice values go through the crate's own
  `decode_pdf_text_string`. One unit test added (a PDFDocEncoding name and a UTF-16BE value).

Measured on invented filled forms: a German form with umlauts in its field names went from no text at all to
complete text; the crate's own unit tests pass as before (17 tests that need upstream fixture files not
shipped in the crates.io package fail with and without the patch).

## Scans with a few typed lines on top (`leafmind-scan-with-typed-lines.patch`)

A scanned page with a few lines typed over it in a word processor — a certificate or form with a phone number,
email, name and date added — was classified as text (`TextBased`, no page needing OCR), so only the typed
lines were read and the scanned body was lost. pdf-inspector's scan rule wants at most one image (a scan with
a logo or stamp image has more) and almost no or barely varied text (typed lines are varied, and a few lines
in several fonts clear the 10-operator floor).

- `src/detector/content_scan.rs`: the executed content scan also counts the bytes of text painted in a visible
  render mode (not 3 or 7), `visible_text_bytes`.
- `src/detector.rs`: new rule `sparse_text_over_covering_image` — drawn images cover at least half of the page
  and it shows under 200 bytes of visible text (not for a page with an invisible text layer, which has its own
  reason, or with form content left unread past the byte budget). Applied in the three places that decide
  OCR: the template-image count of classification, the per-page routing of `Mixed` documents, and
  `page_ocr_signals` for per-page extraction. Two tests added: a scan with ten typed lines (103 bytes) goes to
  OCR, and a real text page over the same image (12 body lines) stays native.

Measured: a real certificate scan (3 images, 6 typed lines, 85 bytes) went from `TextBased` to
`Mixed` with its page needing OCR; pdf-inspector's own letterhead test (12 lines, about 480 bytes) stays
native; the crate's tests pass as before (same 17 fixture failures).

## Removed from the copy (not needed to build)

`target/`, `Cargo.lock`, the Python stub `pdf_inspector.pyi` and the crates.io packaging markers.

## Updating

Take the new version's source from crates.io, apply `leafmind-keep-zwnj.patch`,
`leafmind-form-field-strings.patch` and `leafmind-scan-with-typed-lines.patch` (`patch -p1` inside this
folder), keep this file, and re-run the ZWNJ check, the form-field test and the typed-lines tests. Drop a patch once upstream fixes the same thing itself.

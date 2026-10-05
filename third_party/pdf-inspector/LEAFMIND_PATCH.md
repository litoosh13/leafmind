# leafmind's patched copy of pdf-inspector 1.24.0

This folder is pdf-inspector 1.24.0 as published on crates.io
(https://crates.io/crates/pdf-inspector/1.24.0, upstream https://github.com/firecrawl/pdf-inspector;
`.crate` SHA-256 `e22dc125a533d212c847c8c85e4fcb7358f4384869ef76b2b8721f039b1b633a`), MIT (see `LICENSE`).
The CMap tables in `external/bcmaps/` are Adobe's, under the licence in `external/bcmaps/LICENSE`.
It is used through `[patch.crates-io]` in the workspace `Cargo.toml`.

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

## Removed from the copy (not needed to build)

`target/`, `Cargo.lock`, the Python stub `pdf_inspector.pyi` and the crates.io packaging markers.

## Updating

Take the new version's source from crates.io, apply `leafmind-keep-zwnj.patch` and
`leafmind-form-field-strings.patch` (`patch -p1` inside this folder), keep this file, and re-run the ZWNJ
check and the form-field test. Drop a patch once upstream fixes the same thing itself.

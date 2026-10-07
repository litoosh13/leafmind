//! A prepared document as bytes, so an app can keep it (e.g. in a cache) and skip indexing next time.
//! Kept: the chunks, their embeddings (the slow part) and the language; the keyword index is rebuilt on load.
//!
//! Layout, little-endian: `LMQA`, format (u32), embedder fingerprint (u64), chunking (u8), language (u8, 0 =
//! none), embedding size (u32), chunk count (u32), then per chunk: page (u32), text, section flag (u8) and
//! section, embedding (f32 each). Texts are a u32 byte length and UTF-8.

use crate::language::Language;
use crate::text::Chunking;
use crate::{Error, Passage};

const MAGIC: &[u8; 4] = b"LMQA";
/// Raise when the layout or the meaning of a saved field changes.
const FORMAT: u32 = 1;

/// What a saved document depends on: it can only be used with the same embedder and chunking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    pub embedder: u64,
    pub chunking: Chunking,
}

/// FNV-1a over 8-byte words of the given files (stable across Rust versions, unlike `DefaultHasher`). Reads
/// 1 MB at a time, so the 340 MB embedder adds no memory at load (and about 0.08 s on an Apple M4).
pub(crate) fn fingerprint(files: &[std::path::PathBuf]) -> Result<u64, Error> {
    use std::io::Read;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |w: u64| h = (h ^ w).wrapping_mul(0x0000_0100_0000_01b3);
    let mut buf = vec![0u8; 1 << 20];
    for file in files {
        let io = |e: std::io::Error| Error::Model(format!("{}: {e}", file.display()));
        let mut f = std::fs::File::open(file).map_err(io)?;
        let mut len = 0u64;
        loop {
            // Fill the buffer completely, so only the file's last piece can end inside a word.
            let mut n = 0;
            while n < buf.len() {
                match f.read(&mut buf[n..]).map_err(io)? {
                    0 => break,
                    k => n += k,
                }
            }
            let (words, rest) = buf[..n].as_chunks::<8>();
            words.iter().for_each(|w| mix(u64::from_le_bytes(*w)));
            rest.iter().for_each(|&b| mix(b as u64));
            len += n as u64;
            if n < buf.len() {
                break;
            }
        }
        mix(len);
    }
    Ok(h)
}

fn chunking_code(c: Chunking) -> u8 {
    match c {
        Chunking::Paragraph => 1,
        Chunking::HeadingParagraph => 2,
    }
}

fn language_code(l: Option<Language>) -> u8 {
    match l {
        None => 0,
        Some(Language::English) => 1,
        Some(Language::German) => 2,
        Some(Language::Persian) => 3,
        Some(Language::Arabic) => 4,
    }
}

pub(crate) fn to_bytes(
    key: Key,
    chunks: &[Passage],
    vectors: &[Vec<f32>],
    language: Option<Language>,
) -> Vec<u8> {
    let dim = vectors.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(32 + chunks.len() * (dim * 4 + 200));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&key.embedder.to_le_bytes());
    out.push(chunking_code(key.chunking));
    out.push(language_code(language));
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    let text = |out: &mut Vec<u8>, s: &str| {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    for (chunk, vector) in chunks.iter().zip(vectors) {
        out.extend_from_slice(&chunk.page.to_le_bytes());
        text(&mut out, &chunk.text);
        match &chunk.section {
            Some(s) => {
                out.push(1);
                text(&mut out, s);
            }
            None => out.push(0),
        }
        for x in vector {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Reads bytes from [`to_bytes`]; anything cut off, damaged, of another format or made with another embedder or
/// chunking is an error.
#[allow(clippy::type_complexity)]
pub(crate) fn from_bytes(
    key: Key,
    bytes: &[u8],
) -> Result<(Vec<Passage>, Vec<Vec<f32>>, Option<Language>), Error> {
    let bad = |why: &str| Error::SavedDocument(why.to_string());
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return Err(bad("not a saved leafmind document"));
    }
    let format = r.u32()?;
    if format != FORMAT {
        return Err(Error::SavedDocument(format!(
            "saved in format {format}, this leafmind reads format {FORMAT}"
        )));
    }
    if r.u64()? != key.embedder {
        return Err(bad("saved with another embedder model"));
    }
    if r.u8()? != chunking_code(key.chunking) {
        return Err(bad("saved with another chunking setting"));
    }
    let language = match r.u8()? {
        0 => None,
        1 => Some(Language::English),
        2 => Some(Language::German),
        3 => Some(Language::Persian),
        4 => Some(Language::Arabic),
        _ => return Err(bad("unknown language")),
    };
    let dim = r.u32()? as usize;
    let count = r.u32()? as usize;
    // Each chunk takes at least 9 bytes plus its embedding: refuse counts the bytes cannot hold before allocating.
    if count.saturating_mul(dim.saturating_mul(4).saturating_add(9)) > r.0.len() {
        return Err(bad("cut off"));
    }
    let (mut chunks, mut vectors) = (Vec::with_capacity(count), Vec::with_capacity(count));
    for _ in 0..count {
        let page = r.u32()?;
        let text = r.text()?;
        let section = match r.u8()? {
            0 => None,
            1 => Some(r.text()?),
            _ => return Err(bad("damaged")),
        };
        chunks.push(Passage {
            page,
            text,
            section,
        });
        let raw = r.take(dim * 4)?;
        vectors.push(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        );
    }
    if !r.0.is_empty() {
        return Err(bad("damaged (bytes left over)"));
    }
    Ok((chunks, vectors, language))
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if n > self.0.len() {
            return Err(Error::SavedDocument("cut off".into()));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn text(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| Error::SavedDocument("damaged text".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: Key = Key {
        embedder: 42,
        chunking: Chunking::Paragraph,
    };

    fn sample() -> (Vec<Passage>, Vec<Vec<f32>>) {
        let p = |page, text: &str, section: Option<&str>| Passage {
            page,
            text: text.into(),
            section: section.map(Into::into),
        };
        (
            vec![
                p(1, "Garden rules", None),
                p(
                    2,
                    "The rent is 120 Euro. Wasser ist inklusive.",
                    Some("Rent"),
                ),
            ],
            vec![vec![0.6, -0.8, 0.0], vec![0.0, 1.0, -0.0]],
        )
    }

    #[test]
    fn round_trip() {
        let (chunks, vectors) = sample();
        let bytes = to_bytes(KEY, &chunks, &vectors, Some(Language::German));
        assert_eq!(
            from_bytes(KEY, &bytes).unwrap(),
            (chunks, vectors, Some(Language::German))
        );
        let empty = to_bytes(KEY, &[], &[], None);
        assert_eq!(from_bytes(KEY, &empty).unwrap(), (vec![], vec![], None));
    }

    #[test]
    fn refuses_what_it_cannot_use() {
        let (chunks, vectors) = sample();
        let bytes = to_bytes(KEY, &chunks, &vectors, None);
        let err = |key, b: &[u8]| from_bytes(key, b).unwrap_err().to_string();
        assert!(err(Key { embedder: 7, ..KEY }, &bytes).contains("another embedder"));
        assert!(
            err(
                Key {
                    chunking: Chunking::HeadingParagraph,
                    ..KEY
                },
                &bytes
            )
            .contains("another chunking")
        );
        let mut old = bytes.clone();
        old[4] = 0;
        assert!(err(KEY, &old).contains("format 0"));
        assert!(err(KEY, b"%PDF-1.7").contains("not a saved"));
        // Every cut is refused, never a panic.
        for n in 0..bytes.len() {
            assert!(from_bytes(KEY, &bytes[..n]).is_err(), "cut at {n}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(err(KEY, &longer).contains("left over"));
        // A huge chunk count in a short file is refused before anything is allocated.
        let mut huge = bytes;
        huge[22..26].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(err(KEY, &huge).contains("cut off"));
    }

    #[test]
    fn fingerprint_follows_the_bytes() {
        let dir = std::env::temp_dir().join("leafmind-qa-fingerprint");
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::write(&a, b"model bytes, version 1").unwrap();
        std::fs::write(&b, b"model bytes, version 2").unwrap();
        let fa = fingerprint(std::slice::from_ref(&a)).unwrap();
        assert_eq!(fa, fingerprint(std::slice::from_ref(&a)).unwrap());
        assert_ne!(fa, fingerprint(std::slice::from_ref(&b)).unwrap());
        // Fixed across Rust versions and machines.
        assert_eq!(fa, 0x2209_b749_56d9_2c1c, "{fa:#x}");
    }
}

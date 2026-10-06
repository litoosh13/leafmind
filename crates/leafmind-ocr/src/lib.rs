//! Reads the text of scanned pages with Tesseract 5, on device, in English, German, Persian and Arabic.
//!
//! The app ships the Tesseract library and its language files (`tessdata`: `eng`, `deu`, `fas`, `ara`, and
//! `osd` for script detection; see THIRD_PARTY.md) and gives leafmind their paths; nothing is linked at build
//! time. Pages come in as images (RGBA, like leafmind-fields), so any PDF renderer can be used. The text comes
//! out with paragraphs separated by a blank line, ready for `leafmind_qa::QaEngine::index_pages`.
//!
//! ```no_run
//! use leafmind_ocr::{OcrEngine, OcrLanguage, OcrModels};
//! let ocr = OcrEngine::load(&OcrModels {
//!     tesseract: "lib/libtesseract.5.dylib".into(),
//!     tessdata: "models/tessdata".into(),
//! })?;
//! # let (rgba, width, height) = (vec![255u8; 4 * 100 * 100], 100, 100);
//! // Which languages to read with, from the page's script (when the document language is not known).
//! let languages = match ocr.detect_script(&rgba, width, height, Some(300))? {
//!     Some(script) => script.languages(),
//!     None => &[OcrLanguage::English, OcrLanguage::German],
//! };
//! let page = ocr.read(&rgba, width, height, languages, Some(300))?;
//! println!("{} (confidence {:.2})", page.text, page.confidence);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Mutex;

mod layout;
mod tesseract;

/// What can go wrong.
#[derive(Debug)]
pub enum Error {
    /// The Tesseract library or its language files could not be loaded or used (the message says why).
    Tesseract(String),
    /// The pixels do not match width × height × 4.
    ImageSize,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tesseract(why) => write!(f, "Tesseract: {why}"),
            Error::ImageSize => write!(f, "the image data does not match its width and height"),
        }
    }
}

impl std::error::Error for Error {}

/// Languages leafmind can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OcrLanguage {
    English,
    German,
    Persian,
    Arabic,
}

impl OcrLanguage {
    /// Tesseract's name for the language file (`<code>.traineddata`).
    pub fn code(self) -> &'static str {
        match self {
            OcrLanguage::English => "eng",
            OcrLanguage::German => "deu",
            OcrLanguage::Persian => "fas",
            OcrLanguage::Arabic => "ara",
        }
    }
}

/// The writing system of a page, as Tesseract's script detection sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    Latin,
    Arabic,
}

impl Script {
    /// The languages to read a page of this script with.
    pub fn languages(self) -> &'static [OcrLanguage] {
        match self {
            Script::Latin => &[OcrLanguage::English, OcrLanguage::German],
            Script::Arabic => &[OcrLanguage::Persian, OcrLanguage::Arabic],
        }
    }
}

/// Where the app keeps Tesseract.
#[derive(Clone, Debug)]
pub struct OcrModels {
    /// The Tesseract 5 library: `libtesseract.5.dylib`, `libtesseract.so.5` or the Windows DLL.
    pub tesseract: PathBuf,
    /// Folder with the `.traineddata` language files.
    pub tessdata: PathBuf,
}

/// The text of one page.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrPage {
    /// Paragraphs separated by a blank line, lines of a paragraph joined by spaces.
    pub text: String,
    /// Median word confidence (0–1); low values mean a poor scan or the wrong languages.
    pub confidence: f32,
    /// The recognised words with their positions, e.g. for a searchable text layer over the scan. In
    /// Tesseract's order, line by line; `text` follows it except for form grids, which it reads row by row.
    pub words: Vec<OcrWord>,
}

/// One recognised word.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrWord {
    pub text: String,
    /// Left, top, right, bottom in pixels of the image passed to [`OcrEngine::read`].
    pub bounds: [f32; 4],
    /// Tesseract's confidence for this word, 0–1.
    pub confidence: f32,
    /// The line it is on, counted from 0 in Tesseract's order; the words of a line share it and come in the
    /// order Tesseract gives them (right-to-left lines too).
    pub line: u32,
}

/// Loaded Tesseract. Can be shared between threads; each language set gets its own Tesseract instance.
pub struct OcrEngine {
    api: Box<tesseract::Api>,
    tessdata: PathBuf,
    // The instances borrow `api` (boxed, so its address never changes); `Drop` frees them first.
    instances: Mutex<HashMap<String, tesseract::Instance<'static>>>,
}

impl Drop for OcrEngine {
    fn drop(&mut self) {
        // The instances borrow `api`: free them before it.
        self.instances
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// RGBA pixels to one grey byte per pixel.
fn grey(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    if rgba.len() != width as usize * height as usize * 4 {
        return Err(Error::ImageSize);
    }
    Ok(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8)
        .collect())
}

impl OcrEngine {
    /// Loads the Tesseract library. Language files are read when first used.
    pub fn load(models: &OcrModels) -> Result<Self, Error> {
        Ok(OcrEngine {
            api: Box::new(tesseract::Api::load(&models.tesseract)?),
            tessdata: models.tessdata.clone(),
            instances: Mutex::new(HashMap::new()),
        })
    }

    /// The Tesseract version, e.g. "5.5.0".
    pub fn tesseract_version(&self) -> &str {
        &self.api.version
    }

    fn with_instance<T>(
        &self,
        languages: &str,
        run: impl FnOnce(&mut tesseract::Instance<'static>) -> T,
    ) -> Result<T, Error> {
        let mut instances = self.instances.lock().unwrap_or_else(|e| e.into_inner());
        if !instances.contains_key(languages) {
            // Safety: `api` is boxed (fixed address) and outlives every instance (see Drop).
            let api: &'static tesseract::Api =
                unsafe { &*(self.api.as_ref() as *const tesseract::Api) };
            instances.insert(
                languages.to_string(),
                tesseract::Instance::new(api, &self.tessdata, languages)?,
            );
        }
        Ok(run(instances.get_mut(languages).unwrap()))
    }

    /// Reads a page image (RGBA) in the given languages. `dpi` is the scan resolution if known (better
    /// results; Tesseract guesses otherwise).
    pub fn read(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        languages: &[OcrLanguage],
        dpi: Option<u32>,
    ) -> Result<OcrPage, Error> {
        let mut grey = grey(rgba, width, height)?;
        clean_shading(&mut grey, width as usize, height as usize);
        let codes: Vec<&str> = languages.iter().map(|l| l.code()).collect();
        let tsv = self.with_instance(&codes.join("+"), |t| t.tsv(&grey, width, height, dpi))??;
        let (text, confidence, words) = layout::page_text(&tsv);
        Ok(OcrPage {
            text,
            confidence,
            words,
        })
    }

    /// The page's language: its script (see [`Self::detect_script`]), then one reading with the script's first
    /// language, then the letters and words only one of the two languages uses — Persian پ چ ژ گ against
    /// Arabic ة; German ä ö ü ß and common German words against common English words. Reading with the one
    /// right language is clearly better than reading with both (e.g. Persian 3.9 % character errors instead
    /// of 13 %), so call this on a page with plenty of text and read the document with the result.
    /// `None` when the script is not Latin or Arabic.
    pub fn detect_language(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        dpi: Option<u32>,
    ) -> Result<Option<OcrLanguage>, Error> {
        let Some(script) = self.detect_script(rgba, width, height, dpi)? else {
            return Ok(None);
        };
        let [first, second] = script.languages() else {
            unreachable!()
        };
        let text = self.read(rgba, width, height, &[*first], dpi)?.text;
        Ok(Some(if language_markers(&text, script) {
            *second
        } else {
            *first
        }))
    }

    /// The page's script, when Tesseract is reasonably sure (needs `osd.traineddata`). Pages with too little
    /// text give `None`.
    pub fn detect_script(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        dpi: Option<u32>,
    ) -> Result<Option<Script>, Error> {
        let grey = grey(rgba, width, height)?;
        let found = self.with_instance("osd", |t| {
            t.orientation_and_script(&grey, width, height, dpi)
        })?;
        Ok(match found {
            Some((_, _, name, confidence)) if confidence > 1.0 => match name.as_str() {
                "Latin" => Some(Script::Latin),
                "Arabic" => Some(Script::Arabic),
                _ => None,
            },
            _ => None,
        })
    }
}

/// Grey shading printed as a dot pattern (a halftone, e.g. behind table rows of an invoice) makes Tesseract drop
/// the text on it. Such areas are found as many isolated dark dots (at most one dark neighbour) that a 3×3
/// median removes, densely packed (more than 6 % of a 25-pixel window), and only there the page is smoothed
/// with that median, twice; the rest stays as scanned (smoothing whole pages costs real scans much more than it
/// gains). A page needs at least 0.2 % of such dense spots, so scanner speckle does not count; thin text strokes
/// are not isolated dots, also at low resolutions. Measured: the shaded totals table of a real invoice scan was
/// lost and is read with it; 62 benchmark scans and 18 other real scans are unchanged.
fn clean_shading(g: &mut [u8], w: usize, h: usize) {
    if w < 3 || h < 3 {
        return;
    }
    let med = median3(g, w, h);
    let dark = |x: usize, y: usize| g[y * w + x] < 128;
    let mut dots = vec![0u32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if dark(x, y) && med[y * w + x] >= 128 {
                let neighbours = (y - 1..=y + 1)
                    .flat_map(|yy| (x - 1..=x + 1).map(move |xx| (xx, yy)))
                    .filter(|&(xx, yy)| (xx, yy) != (x, y) && dark(xx, yy))
                    .count();
                dots[y * w + x] = u32::from(neighbours <= 1);
            }
        }
    }
    let dense = window_sums(&dots, w, h, 12);
    let seeds: Vec<u32> = dense
        .iter()
        .map(|&(sum, area)| u32::from(sum as f32 > 0.06 * area as f32))
        .collect();
    if (seeds.iter().sum::<u32>() as f32) < 0.002 * (w * h) as f32 {
        return;
    }
    let near = window_sums(&seeds, w, h, 30);
    let med2 = median3(&med, w, h);
    for (i, (sum, _)) in near.iter().enumerate() {
        if *sum > 0 {
            g[i] = med2[i];
        }
    }
}

/// For every pixel, the sum of `v` over the square window of radius `r` around it (cut at the edges) and that
/// window's area, from an integral image.
fn window_sums(v: &[u32], w: usize, h: usize, r: usize) -> Vec<(u32, u32)> {
    let mut ii = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0;
        for x in 0..w {
            row += v[y * w + x];
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let sum = ii[y1 * (w + 1) + x1] + ii[y0 * (w + 1) + x0]
                - ii[y0 * (w + 1) + x1]
                - ii[y1 * (w + 1) + x0];
            out.push((sum, ((x1 - x0) * (y1 - y0)) as u32));
        }
    }
    out
}

/// 3×3 median filter; edge pixels keep their value.
fn median3(g: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = g.to_vec();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut v = [0u8; 9];
            for dy in 0..3 {
                v[dy * 3..dy * 3 + 3]
                    .copy_from_slice(&g[(y + dy - 1) * w + x - 1..(y + dy - 1) * w + x + 2]);
            }
            v.sort_unstable();
            out[y * w + x] = v[4];
        }
    }
    out
}

/// Common words that mark German or English text.
const GERMAN_WORDS: [&str; 14] = [
    "der", "die", "das", "und", "ist", "nicht", "mit", "für", "von", "zu", "den", "des", "ein",
    "eine",
];
const ENGLISH_WORDS: [&str; 13] = [
    "the", "and", "of", "to", "is", "in", "for", "with", "by", "be", "this", "that", "shall",
];

/// Whether `text` shows more signs of the script's second language (Arabic, German) than of its first
/// (Persian, English).
fn language_markers(text: &str, script: Script) -> bool {
    match script {
        Script::Arabic => {
            text.matches('ة').count() > text.chars().filter(|c| "پچژگ".contains(*c)).count()
        }
        Script::Latin => {
            let lower = text.to_lowercase();
            let words: Vec<&str> = lower.split(|c: char| !c.is_alphabetic()).collect();
            let german = words.iter().filter(|w| GERMAN_WORDS.contains(w)).count()
                + lower.chars().filter(|c| "äöüß".contains(*c)).count();
            let english = words.iter().filter(|w| ENGLISH_WORDS.contains(w)).count();
            german > english
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_dot_pattern_shading_is_smoothed() {
        let (w, h) = (600, 600);
        let mut page = vec![255u8; w * h];
        // A shaded box: one dark dot every second pixel (a halftone).
        for y in (100..300).step_by(2) {
            for x in (100..500).step_by(2) {
                page[y * w + x] = 0;
            }
        }
        // A stroke of text elsewhere.
        for y in 450..454 {
            for x in 100..400 {
                page[y * w + x] = 0;
            }
        }
        let mut cleaned = page.clone();
        clean_shading(&mut cleaned, w, h);
        let dark = |g: &[u8], ys: std::ops::Range<usize>| {
            ys.flat_map(|y| (100..500).map(move |x| (y, x)))
                .filter(|&(y, x)| g[y * w + x] < 128)
                .count()
        };
        assert!(dark(&page, 120..280) > 10_000 && dark(&cleaned, 120..280) == 0);
        assert_eq!(cleaned[450 * w..454 * w], page[450 * w..454 * w]);
        // A few dots (scanner speckle) are not a shaded area: the page stays as it is.
        let mut speckle = vec![255u8; w * h];
        for y in (100..110).step_by(2) {
            for x in (100..110).step_by(2) {
                speckle[y * w + x] = 0;
            }
        }
        let before = speckle.clone();
        clean_shading(&mut speckle, w, h);
        assert_eq!(speckle, before);
    }

    #[test]
    fn grey_conversion_and_size_check() {
        assert_eq!(
            grey(&[255, 255, 255, 255, 0, 0, 0, 255], 2, 1).unwrap(),
            [255, 0]
        );
        assert!(matches!(grey(&[0; 7], 2, 1), Err(Error::ImageSize)));
    }

    #[test]
    fn marker_letters_and_words() {
        assert!(language_markers(
            "المدة الزمنية للعقد سنة واحدة",
            Script::Arabic
        ));
        assert!(!language_markers(
            "مدت قرارداد یک سال است و پرداخت ماهانه انجام می شود",
            Script::Arabic
        ));
        assert!(language_markers(
            "Die Pacht für den Garten ist im März fällig.",
            Script::Latin
        ));
        assert!(!language_markers(
            "The rent for the garden is due in March.",
            Script::Latin
        ));
    }

    #[test]
    fn scripts_choose_languages() {
        assert_eq!(
            Script::Arabic.languages(),
            [OcrLanguage::Persian, OcrLanguage::Arabic]
        );
        assert_eq!(OcrLanguage::Persian.code(), "fas");
    }

    #[test]
    fn a_missing_library_is_an_error() {
        let models = OcrModels {
            tesseract: "no-such-libtesseract".into(),
            tessdata: "no-such-dir".into(),
        };
        let err = OcrEngine::load(&models).err().unwrap();
        assert!(
            err.to_string()
                .starts_with("Tesseract: no-such-libtesseract"),
            "{err}"
        );
    }
}

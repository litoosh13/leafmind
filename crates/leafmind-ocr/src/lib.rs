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
        let grey = grey(rgba, width, height)?;
        let codes: Vec<&str> = languages.iter().map(|l| l.code()).collect();
        let tsv = self.with_instance(&codes.join("+"), |t| t.tsv(&grey, width, height, dpi))??;
        let (text, confidence) = layout::page_text(&tsv);
        Ok(OcrPage { text, confidence })
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

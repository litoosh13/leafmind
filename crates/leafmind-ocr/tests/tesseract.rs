//! With the real Tesseract library, on the invented library-card form of leafmind-fields' tests.
//! Run: LEAFMIND_TESSERACT=<libtesseract> LEAFMIND_TESSDATA=<tessdata with eng, osd>
//! cargo test -p leafmind-ocr -- --ignored

use leafmind_ocr::{OcrEngine, OcrLanguage, OcrModels, Script};

fn engine() -> OcrEngine {
    OcrEngine::load(&OcrModels {
        tesseract: std::env::var("LEAFMIND_TESSERACT")
            .expect("LEAFMIND_TESSERACT")
            .into(),
        tessdata: std::env::var("LEAFMIND_TESSDATA")
            .expect("LEAFMIND_TESSDATA")
            .into(),
    })
    .unwrap()
}

fn page() -> (Vec<u8>, u32, u32) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../leafmind-fields/tests/data/library-card.png"
    );
    let img = image::open(path).unwrap().to_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

#[test]
#[ignore = "needs the Tesseract library and tessdata (LEAFMIND_TESSERACT, LEAFMIND_TESSDATA)"]
fn reads_an_invented_form() {
    let ocr = engine();
    assert!(
        ocr.tesseract_version().starts_with('5'),
        "{}",
        ocr.tesseract_version()
    );
    let (rgba, w, h) = page();
    // The page is A4 at about 108 dpi.
    let page = ocr
        .read(&rgba, w, h, &[OcrLanguage::English], Some(108))
        .unwrap();
    assert!(
        page.text.contains("Library card application"),
        "{}",
        page.text
    );
    assert!(page.text.contains("Date of birth"), "{}", page.text);
    assert!(page.confidence > 0.8, "{}", page.confidence);
    // Reading again reuses the same Tesseract instance and gives the same text.
    assert_eq!(
        ocr.read(&rgba, w, h, &[OcrLanguage::English], Some(108))
            .unwrap(),
        page
    );
}

#[test]
#[ignore = "needs the Tesseract library and tessdata (LEAFMIND_TESSERACT, LEAFMIND_TESSDATA)"]
fn detects_latin_script() {
    let (rgba, w, h) = page();
    assert_eq!(
        engine().detect_script(&rgba, w, h, Some(108)).unwrap(),
        Some(Script::Latin)
    );
}

#[test]
#[ignore = "needs the Tesseract library and tessdata (LEAFMIND_TESSERACT, LEAFMIND_TESSDATA)"]
fn a_missing_language_is_an_error() {
    let (rgba, w, h) = page();
    // Without a resolution Tesseract estimates one.
    assert!(
        engine()
            .read(&rgba, w, h, &[OcrLanguage::English], None)
            .is_ok()
    );
    let dir = std::env::temp_dir().join("leafmind-ocr-empty-tessdata");
    std::fs::create_dir_all(&dir).unwrap();
    let ocr = OcrEngine::load(&OcrModels {
        tesseract: std::env::var("LEAFMIND_TESSERACT").unwrap().into(),
        tessdata: dir,
    })
    .unwrap();
    let err = ocr
        .read(&rgba, w, h, &[OcrLanguage::English], None)
        .unwrap_err();
    assert!(
        err.to_string().contains("could not load the languages eng"),
        "{err}"
    );
}

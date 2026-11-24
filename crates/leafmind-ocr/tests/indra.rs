//! With the real Indra models, on invented pages: the library-card form of leafmind-fields' tests (also turned a
//! quarter turn) and an English/Persian notice (tests/data/make_testdata.py).
//! Run: LEAFMIND_ORT=<onnxruntime library> LEAFMIND_INDRA=<indra-ocr model folder>
//! cargo test -p leafmind-ocr --features indra -- --ignored

#![cfg(feature = "indra")]

use leafmind_ocr::{IndraEngine, IndraModels};
use std::sync::OnceLock;

fn engine() -> &'static IndraEngine {
    static ENGINE: OnceLock<IndraEngine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        IndraEngine::load(
            &IndraModels {
                onnxruntime: std::env::var("LEAFMIND_ORT").expect("LEAFMIND_ORT").into(),
                models: std::env::var("LEAFMIND_INDRA")
                    .expect("LEAFMIND_INDRA")
                    .into(),
            },
            4,
        )
        .unwrap()
    })
}

fn image(path: &str) -> image::RgbaImage {
    image::open(format!("{}/{path}", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
        .to_rgba8()
}

#[test]
#[ignore = "needs ONNX Runtime and the Indra models (LEAFMIND_ORT, LEAFMIND_INDRA)"]
fn reads_english_and_persian_on_one_page() {
    let img = image("tests/data/bilingual-notice.png");
    let (w, h) = img.dimensions();
    let page = engine().read(img.as_raw(), w, h).unwrap();
    for expected in [
        "Equipment Loan Notice",
        "Reference: LN-4471",
        "public holidays",
        // Persian in logical order, digits as the page writes them.
        "امانت تجهیزات",
        "۴۴۷۱",
        "تعطیل",
    ] {
        assert!(
            page.text.contains(expected),
            "{expected:?} not in {:?}",
            page.text
        );
    }
    assert!(page.confidence > 0.8, "{}", page.confidence);
}

#[test]
#[ignore = "needs ONNX Runtime and the Indra models (LEAFMIND_ORT, LEAFMIND_INDRA)"]
fn reads_a_form_also_when_the_scan_is_turned() {
    let img = image("../leafmind-fields/tests/data/library-card.png");
    let (w, h) = img.dimensions();
    let upright = engine().read(img.as_raw(), w, h).unwrap();
    assert!(
        upright.text.contains("Library card application"),
        "{}",
        upright.text
    );
    assert!(upright.text.contains("Date of birth"), "{}", upright.text);
    // The word "Library" where make_testdata.py drew it: x 56, baseline 70, 16 pt, at 1.5 pixels per point.
    let word = upright.words.iter().find(|w| w.text == "Library").unwrap();
    assert!((word.bounds[0] - 84.0).abs() < 12.0, "{:?}", word.bounds);
    assert!(
        word.bounds[1] < 105.0 && word.bounds[3] > 100.0,
        "{:?}",
        word.bounds
    );

    // The same page scanned a quarter turn clockwise: read upright, boxes on the image as given.
    let turned = image::imageops::rotate90(&img);
    let page = engine().read(turned.as_raw(), h, w).unwrap();
    assert!(
        page.text.contains("Library card application"),
        "{}",
        page.text
    );
    let word = page.words.iter().find(|w| w.text == "Library").unwrap();
    // Turned clockwise, (x, y) moves to (h − y, x).
    let [l, t, r, b] = word.bounds;
    assert!(
        l < h as f32 - 95.0 && r > h as f32 - 110.0,
        "{:?}",
        word.bounds
    );
    assert!((t - 84.0).abs() < 12.0 && b > t, "{:?}", word.bounds);
}

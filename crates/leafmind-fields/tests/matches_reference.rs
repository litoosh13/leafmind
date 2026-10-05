//! The detector must find exactly the fields of the reference run (Python: Pillow 12.3 letterbox + ONNX
//! Runtime, the model card's recipe) on an invented, generic test form (tests/data, made by
//! tests/data/make_testdata.py): same number, same kinds, boxes overlapping one to one (IoU >= 0.9).
//! `end_to_end` starts from the page image; `given_same_input` starts from the reference's own 640×640
//! model input, so a failure there points at the model run or the decoding, not the scaling.

use leafmind_fields::{Field, FieldFinder, FieldKind, Options, overlap};

/// The reference run used the model card's recipe (threshold 0.3, overlap 0.6, the whole page only), not
/// leafmind's default.
const CARD: Options = Options {
    min_score: 0.3,
    max_overlap: 0.6,
    tiles: false,
};

fn root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn finder() -> FieldFinder {
    let model = std::fs::read(format!("{}/../../models/form-field-v1-nano.onnx", root())).unwrap();
    FieldFinder::from_onnx(&model).unwrap()
}

fn expected() -> serde_json::Map<String, serde_json::Value> {
    let text = std::fs::read_to_string(format!("{}/tests/data/expected.json", root())).unwrap();
    serde_json::from_str::<serde_json::Value>(&text)
        .unwrap()
        .as_object()
        .unwrap()
        .clone()
}

fn assert_same(page: &str, found: &[Field], reference: &serde_json::Value) {
    let reference = reference["fields"].as_array().unwrap();
    assert_eq!(found.len(), reference.len(), "{page}: number of fields");
    let mut used = vec![false; found.len()];
    for r in reference {
        let kind = match r["kind"].as_str().unwrap() {
            "text" => FieldKind::Text,
            "choice" => FieldKind::Choice,
            _ => FieldKind::Signature,
        };
        let b: Vec<f32> = r["box"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let b = [b[0], b[1], b[2], b[3]];
        let hit = (0..found.len())
            .find(|&i| !used[i] && found[i].kind == kind && overlap(&found[i].bounds, &b) >= 0.9);
        assert!(hit.is_some(), "{page}: reference field {r} not found");
        used[hit.unwrap()] = true;
    }
}

#[test]
fn end_to_end() {
    let finder = finder();
    for (page, reference) in expected() {
        let image = image::open(format!("{}/tests/data/{page}.png", root()))
            .unwrap()
            .to_rgba8();
        let (w, h) = image.dimensions();
        assert_same(
            &page,
            &finder.find_with(image.as_raw(), w, h, CARD).unwrap(),
            &reference,
        );
    }
}

#[test]
fn given_same_input() {
    let finder = finder();
    for (page, reference) in expected() {
        let input = image::open(format!("{}/tests/data/{page}.model-input.png", root()))
            .unwrap()
            .to_rgb8();
        let (w, h) = (
            reference["width"].as_f64().unwrap() as f32,
            reference["height"].as_f64().unwrap() as f32,
        );
        let scale = (640.0 / w).min(640.0 / h);
        assert_same(
            &page,
            &finder.find_prepared(input.as_raw(), scale, CARD).unwrap(),
            &reference,
        );
    }
}

#[test]
fn tells_filled_from_empty() {
    let state = std::fs::read(format!("{}/../../models/form-field-v1-state.onnx", root())).unwrap();
    let finder = finder().with_state(&state).unwrap();
    let mut image = image::open(format!("{}/tests/data/library-card.png", root()))
        .unwrap()
        .to_rgba8();
    let (w, h) = image.dimensions();
    let blank = finder.find(image.as_raw(), w, h).unwrap();
    assert!(blank.iter().all(|f| f.filled == Some(false)), "{blank:?}");
    // Tick the first choice button and write into the widest text field (thick dark strokes).
    let tick = blank
        .iter()
        .find(|f| f.kind == FieldKind::Choice)
        .unwrap()
        .bounds;
    let text = blank
        .iter()
        .filter(|f| f.kind == FieldKind::Text)
        .max_by(|a, b| (a.bounds[2] - a.bounds[0]).total_cmp(&(b.bounds[2] - b.bounds[0])))
        .unwrap()
        .bounds;
    let mut stroke = |x: f32, y: f32| {
        for dy in -1..=1 {
            for dx in -1..=1 {
                image.put_pixel(
                    (x as i32 + dx) as u32,
                    (y as i32 + dy) as u32,
                    image::Rgba([20, 20, 20, 255]),
                );
            }
        }
    };
    for i in 0..=40 {
        let t = i as f32 / 40.0;
        let (x0, y0, x1, y1) = (tick[0] + 3.0, tick[1] + 3.0, tick[2] - 3.0, tick[3] - 3.0);
        stroke(x0 + t * (x1 - x0), y0 + t * (y1 - y0));
        stroke(x0 + t * (x1 - x0), y1 - t * (y1 - y0));
    }
    // A wavy line of "handwriting" over the left half of the text field.
    let (x0, mid, len) = (
        text[0] + 6.0,
        (text[1] + text[3]) / 2.0,
        (text[2] - text[0]) / 2.0,
    );
    for i in 0..=400 {
        let x = x0 + len * i as f32 / 400.0;
        stroke(x, mid + 0.25 * (text[3] - text[1]) * (x / 4.0).sin());
    }
    let filled = finder.find(image.as_raw(), w, h).unwrap();
    for (bounds, want) in [(tick, true), (text, true)] {
        let f = filled
            .iter()
            .find(|f| overlap(&f.bounds, &bounds) >= 0.5)
            .unwrap();
        assert_eq!(f.filled, Some(want), "{f:?}");
    }
    let untouched = filled
        .iter()
        .filter(|f| overlap(&f.bounds, &tick) < 0.1 && overlap(&f.bounds, &text) < 0.1);
    assert!(
        untouched.clone().all(|f| f.filled == Some(false)),
        "{:?}",
        untouched.collect::<Vec<_>>()
    );
}

#[test]
fn rejects_wrong_image_size() {
    assert!(finder().find(&[0u8; 10], 2, 2).is_err());
}

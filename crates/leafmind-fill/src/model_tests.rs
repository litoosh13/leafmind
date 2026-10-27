//! Tests with the real models, on one invented club membership form (words and boxes made here).
//!
//! Run: `LEAFMIND_QA_MODELS=<dir with gte-embed/> LEAFMIND_ORT=<onnxruntime library>
//! LEAFMIND_NEEDLE=<libneedle3 library> LEAFMIND_FILL_MODEL=<fill.cact> cargo test -p leafmind-fill --release --
//! --ignored` (gte: scripts/fetch-qa-models.sh; fill.cact: THIRD_PARTY.md).

use crate::{
    Change, FieldKind, FillEngine, FillModels, Form, FormField, PageWords, Proposal, Word,
};
use std::sync::OnceLock;

/// Words of `text` from `x` on line `y`, about 6 points per character.
fn words(text: &str, mut x: f32, y: f32, out: &mut Vec<Word>) {
    for w in text.split(' ') {
        let r = x + 6.0 * w.chars().count() as f32;
        out.push(Word {
            text: w.into(),
            bounds: [x, y, r, y + 10.0],
        });
        x = r + 4.0;
    }
}

fn setup() -> &'static (FillEngine, Form) {
    static S: OnceLock<(FillEngine, Form)> = OnceLock::new();
    S.get_or_init(|| {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let engine = FillEngine::load(&FillModels {
            onnxruntime: env("LEAFMIND_ORT").into(),
            embedder: std::path::Path::new(&env("LEAFMIND_QA_MODELS")).join("gte-embed"),
            needle_library: env("LEAFMIND_NEEDLE").into(),
            needle_model: env("LEAFMIND_FILL_MODEL").into(),
        })
        .unwrap();
        let mut w = vec![];
        let mut fields = vec![];
        words("Garden Club Membership Application", 50.0, 40.0, &mut w);
        let text = [
            "Full name:",
            "Date of birth:",
            "Mobile phone number:",
            "Email address:",
            "Home address:",
            "Mailing address:",
            "Postcode and city:",
            "Date:",
        ];
        for (i, label) in text.iter().enumerate() {
            let y = 80.0 + 30.0 * i as f32;
            words(label, 50.0, y, &mut w);
            fields.push(FormField {
                id: i as u32,
                page: 1,
                bounds: [180.0, y - 2.0, 500.0, y + 12.0],
                kind: FieldKind::Text,
                tooltip: None,
                read_only: false,
            });
        }
        for (i, label) in [
            "I would like the monthly newsletter",
            "I agree to the club rules",
        ]
        .iter()
        .enumerate()
        {
            let y = 330.0 + 30.0 * i as f32;
            words(label, 70.0, y, &mut w);
            fields.push(FormField {
                id: 8 + i as u32,
                page: 1,
                bounds: [50.0, y, 60.0, y + 10.0],
                kind: FieldKind::Checkbox,
                tooltip: None,
                read_only: false,
            });
        }
        words("Membership type:", 50.0, 400.0, &mut w);
        fields.push(FormField {
            id: 10,
            page: 1,
            bounds: [180.0, 398.0, 300.0, 412.0],
            kind: FieldKind::Choice {
                options: vec!["Single".into(), "Family".into(), "Senior".into()],
            },
            tooltip: None,
            read_only: false,
        });
        words("Member number (office use only):", 50.0, 440.0, &mut w);
        fields.push(FormField {
            id: 11,
            page: 1,
            bounds: [250.0, 438.0, 500.0, 452.0],
            kind: FieldKind::Text,
            tooltip: None,
            read_only: true,
        });
        let form = engine
            .prepare(&fields, &[PageWords { page: 1, words: w }])
            .unwrap();
        (engine, form)
    })
}

fn propose(command: &str) -> Proposal {
    let (engine, form) = setup();
    engine.propose(form, command, "10/27/2026").unwrap()
}

/// One action on `field` with `change`.
fn expect(command: &str, field: u32, change: Change) {
    match propose(command) {
        Proposal::Actions(a) if a.len() == 1 && a[0].field == field && a[0].change == change => {}
        other => panic!("{command:?}: expected field {field} {change:?}, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the models"]
fn fills_ticks_and_chooses() {
    expect(
        "my name is Robin Example",
        0,
        Change::Fill("Robin Example".into()),
    );
    expect("phone 0151 2345678", 2, Change::Fill("0151 2345678".into()));
    expect(
        "my email is robin@example.org",
        3,
        Change::Fill("robin@example.org".into()),
    );
    expect(
        "date of birth 12.03.1990",
        1,
        Change::Fill("12.03.1990".into()),
    );
    expect("tick the newsletter", 8, Change::Check(true));
    expect("tick I agree to the club rules", 9, Change::Check(true));
    expect("untick the newsletter", 8, Change::Check(false));
    expect(
        "membership type: Family",
        10,
        Change::Choose("Family".into()),
    );
}

#[test]
#[ignore = "needs the models"]
fn several_fields_in_one_sentence() {
    match propose("name Robin Example, mobile 0170 1112223") {
        Proposal::Actions(a)
            if a.len() == 2
                && (a[0].field, &a[0].change) == (0, &Change::Fill("Robin Example".into()))
                && (a[1].field, &a[1].change) == (2, &Change::Fill("0170 1112223".into())) => {}
        other => panic!("two fields: {other:?}"),
    }
    // One value with a comma stays one.
    expect(
        "home address: 12 Oak Street, Apt 4",
        4,
        Change::Fill("12 Oak Street, Apt 4".into()),
    );
}

#[test]
#[ignore = "needs the models"]
fn other_actions() {
    expect(
        "mailing address same as home address",
        5,
        Change::CopyFrom(4),
    );
    expect("clear the phone number", 2, Change::Clear);
    assert!(matches!(
        propose("sign the form"),
        Proposal::NotSupported { .. }
    ));
    // Today's date as the app writes it; the date field first or among the alternatives.
    match propose("the date is today") {
        Proposal::Actions(a)
            if a[0].change == Change::Fill("10/27/2026".into())
                && (a[0].field == 7 || a[0].alternatives.contains(&7)) => {}
        other => panic!("today: {other:?}"),
    }
    // Read-only fields are never proposed.
    match propose("member number M-1234") {
        Proposal::Actions(a) => assert!(
            a.iter()
                .all(|a| a.field != 11 && !a.alternatives.contains(&11))
        ),
        other => panic!("member number: {other:?}"),
    }
}

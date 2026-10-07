//! Tests with the real models, on one invented garden-plot agreement in four languages. English and
//! German go through a generated PDF; Persian and Arabic are given as page text (the test PDFs have no
//! Arabic-script font). Numbers are checked exactly as the document writes them.
//!
//! Run: `LEAFMIND_QA_MODELS=<dir with gte-embed/ and gte-reranker/> LEAFMIND_ORT=<onnxruntime library>
//! cargo test -p leafmind-qa --release -- --ignored` (models: scripts/fetch-qa-models.sh). If the folder also
//! has `qwen3-reranker/` (`scripts/fetch-qa-models.sh <dir> accurate`), every check also runs in accurate mode.
//!
//! On an Apple M4 the correct answers score 0.57–0.95 (fast mode) against the 0.5 cutoff. x86 PCs scored the
//! same int8 models up to 0.18 lower in earlier tests; 0.5 was still the best cutoff on an office PC.

use crate::{Answer, Document, Language, QaEngine, QaModels, QaOptions};
use std::sync::OnceLock;

fn engine() -> &'static QaEngine {
    static ENGINE: OnceLock<QaEngine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let dir = std::path::PathBuf::from(
            std::env::var("LEAFMIND_QA_MODELS").expect("LEAFMIND_QA_MODELS"),
        );
        let accurate = dir.join("qwen3-reranker");
        let models = QaModels {
            onnxruntime: std::env::var("LEAFMIND_ORT").expect("LEAFMIND_ORT").into(),
            embedder: dir.join("gte-embed"),
            reranker: dir.join("gte-reranker"),
            accurate_reranker: accurate.exists().then_some(accurate),
        };
        QaEngine::load(&models, QaOptions::default()).unwrap()
    })
}

/// Fast mode, and accurate mode when its model is there.
fn answers(doc: &Document, question: &str) -> Vec<(&'static str, Answer)> {
    let mut out = vec![("fast", engine().ask(doc, question).unwrap())];
    if let Ok(answer) = engine().ask_accurate(doc, question) {
        out.push(("accurate", answer));
    }
    out
}

/// The answer must be found and one of its sentences must contain `expected`.
fn found(doc: &Document, question: &str, expected: &str) {
    for (mode, answer) in answers(doc, question) {
        match answer {
            Answer::Found { sentences, .. }
                if sentences.iter().any(|s| s.text.contains(expected)) => {}
            other => {
                panic!("{mode} {question:?}: expected an answer with {expected:?}, got {other:?}")
            }
        }
    }
}

fn not_found(doc: &Document, question: &str) {
    for (mode, answer) in answers(doc, question) {
        assert!(
            matches!(answer, Answer::NotFound { .. }),
            "{mode} {question:?}: {answer:?}"
        );
    }
}

fn wrong_language(doc: &Document, question: &str, document: Language, asked: Language) {
    for (mode, answer) in answers(doc, question) {
        assert_eq!(
            answer,
            Answer::WrongLanguage {
                document,
                question: asked
            },
            "{mode} {question:?}"
        );
    }
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn english_pdf() {
    let pdf = crate::test_pdf::pdf(&[
        &[
            "# Garden plot agreement",
            "",
            "The yearly plot rent is 120 Euro and is paid in March.",
            "",
            "Keeping bees is allowed only after written notice to the club.",
        ],
        &[
            "# Water",
            "",
            "Water is included in the rent. The water tap is closed on 31 October.",
            "",
            "The notice period is three months to the end of the garden year.",
            "",
            "Amendment 1: with effect from 2028 the yearly plot rent is 135 Euro.",
        ],
    ]);
    let doc = engine().index_pdf(&pdf).unwrap();
    assert_eq!(doc.language(), Some(Language::English));
    // The amendment is the rent that applies now, so it must be part of the answer.
    found(&doc, "How much is the yearly plot rent?", "135 Euro");
    found(&doc, "When is the water tap closed?", "31 October");
    found(&doc, "How long is the notice period?", "three months");
    not_found(&doc, "What colour is the club house?");
    wrong_language(
        &doc,
        "Wie hoch ist die jährliche Pacht für den Garten?",
        Language::English,
        Language::German,
    );
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn german_pdf() {
    let pdf = crate::test_pdf::pdf(&[
        &[
            "# Kleingarten-Pachtvertrag",
            "",
            "Die jährliche Pacht beträgt 120 Euro und ist im März fällig.",
            "",
            "Bienenhaltung ist nur nach schriftlicher Mitteilung an den Verein erlaubt.",
        ],
        &[
            "# Wasser",
            "",
            "Das Wasser ist in der Pacht enthalten. Der Wasserhahn wird am 31. Oktober geschlossen.",
            "",
            "Die Kündigungsfrist beträgt drei Monate zum Ende des Gartenjahres.",
        ],
    ]);
    let doc = engine().index_pdf(&pdf).unwrap();
    assert_eq!(doc.language(), Some(Language::German));
    found(&doc, "Wie hoch ist die jährliche Pacht?", "120 Euro");
    found(&doc, "Wann wird der Wasserhahn geschlossen?", "31. Oktober");
    found(&doc, "Wie lang ist die Kündigungsfrist?", "drei Monate");
    not_found(&doc, "Welche Farbe hat das Vereinshaus?");
    wrong_language(
        &doc,
        "How long is the notice period for the plot?",
        Language::German,
        Language::English,
    );
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn persian_text() {
    let doc = engine()
        .index_pages(&[
            (
                1,
                "# قرارداد اجاره قطعه باغ\n\nاجاره‌بهای سالانه هر قطعه ۱۲۰ یورو است و در ماه مارس پرداخت می‌شود.\n\nنگهداری زنبور عسل فقط با اطلاع کتبی به انجمن مجاز است.",
            ),
            (
                2,
                "# آب\n\nهزینه آب در اجاره‌بها گنجانده شده است. شیر آب در تاریخ ۳۱ اکتبر بسته می‌شود.\n\nمهلت فسخ قرارداد سه ماه تا پایان سال باغبانی است.",
            ),
        ])
        .unwrap();
    assert_eq!(doc.language(), Some(Language::Persian));
    found(&doc, "اجاره‌بهای سالانه هر قطعه چقدر است؟", "۱۲۰ یورو");
    found(&doc, "شیر آب چه زمانی بسته می‌شود؟", "۳۱ اکتبر");
    found(&doc, "مهلت فسخ قرارداد چقدر است؟", "سه ماه");
    not_found(&doc, "رنگ ساختمان انجمن چیست؟");
    wrong_language(
        &doc,
        "How much is the yearly plot rent?",
        Language::Persian,
        Language::English,
    );
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn arabic_text() {
    let doc = engine()
        .index_pages(&[
            (
                1,
                "# عقد إيجار قطعة حديقة\n\nالإيجار السنوي لكل قطعة ١٢٠ يورو ويدفع في شهر مارس.\n\nيسمح بتربية النحل فقط بعد إخطار الجمعية كتابياً.",
            ),
            (
                2,
                "# الماء\n\nتكلفة الماء مشمولة في الإيجار. يغلق صنبور الماء في ٣١ أكتوبر.\n\nمدة الإشعار لإنهاء العقد ثلاثة أشهر حتى نهاية سنة البستنة.",
            ),
        ])
        .unwrap();
    assert_eq!(doc.language(), Some(Language::Arabic));
    found(&doc, "كم يبلغ الإيجار السنوي لكل قطعة؟", "١٢٠ يورو");
    found(&doc, "متى يغلق صنبور الماء؟", "٣١ أكتوبر");
    found(&doc, "ما هي مدة الإشعار لإنهاء العقد؟", "ثلاثة أشهر");
    not_found(&doc, "ما لون مبنى الجمعية؟");
    wrong_language(
        &doc,
        "Wie lang ist die Kündigungsfrist für den Garten?",
        Language::Arabic,
        Language::German,
    );
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn a_saved_document_answers_the_same() {
    let pdf = crate::test_pdf::pdf(&[&[
        "# Garden plot agreement",
        "",
        "The yearly plot rent is 120 Euro and is paid in March.",
        "",
        "Amendment 1: with effect from 2028 the yearly plot rent is 135 Euro.",
        "",
        "Water is included in the rent. The water tap is closed on 31 October.",
    ]]);
    let doc = engine().index_pdf(&pdf).unwrap();
    let bytes = engine().save_document(&doc);
    let loaded = engine().load_document(&bytes).unwrap();
    assert_eq!(loaded.chunks(), doc.chunks());
    assert_eq!(loaded.language(), doc.language());
    for question in [
        "How much is the yearly plot rent?",
        "When is the water tap closed?",
        "What colour is the club house?",
        "Wie hoch ist die jährliche Pacht?",
    ] {
        // Same answers, sentences and confidences, in both modes.
        assert_eq!(
            answers(&loaded, question),
            answers(&doc, question),
            "{question}"
        );
    }
    // Bytes saved with another embedder model are refused (the fingerprint follows the header's first 8 bytes).
    let mut other = bytes.clone();
    other[8] ^= 1;
    let err = engine().load_document(&other).unwrap_err().to_string();
    assert!(err.contains("another embedder"), "{err}");
}

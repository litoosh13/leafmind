//! Tests with the real models, on one invented garden-plot agreement in four languages. English and
//! German go through a generated PDF; Persian and Arabic are given as page text (the test PDFs have no
//! Arabic-script font). Numbers are checked exactly as the document writes them.
//!
//! Run: `LEAFMIND_QA_MODELS=<dir with gte-embed/ and gte-reranker/> LEAFMIND_ORT=<onnxruntime library>
//! cargo test -p leafmind-qa --release -- --ignored` (models: scripts/fetch-qa-models.sh).
//!
//! On an Apple M4 the correct answers score 0.57–0.95 against the 0.5 cutoff. x86 PCs scored the
//! same int8 models up to 0.18 lower in earlier tests; 0.5 was still the best cutoff on an office PC.

use crate::{Answer, Document, Language, QaEngine, QaModels, QaOptions};
use std::sync::OnceLock;

fn engine() -> &'static QaEngine {
    static ENGINE: OnceLock<QaEngine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let dir = std::path::PathBuf::from(
            std::env::var("LEAFMIND_QA_MODELS").expect("LEAFMIND_QA_MODELS"),
        );
        let models = QaModels {
            onnxruntime: std::env::var("LEAFMIND_ORT").expect("LEAFMIND_ORT").into(),
            embedder: dir.join("gte-embed"),
            reranker: dir.join("gte-reranker"),
        };
        QaEngine::load(&models, QaOptions::default()).unwrap()
    })
}

fn answers(doc: &Document, question: &str) -> Vec<(&'static str, Answer)> {
    vec![("fast", engine().ask(doc, question).unwrap())]
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
        // Same answers, sentences and confidences.
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

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn a_form_with_capital_labels_answers_with_its_rows() {
    // An invented form page as OCR gives it: one row per line, labels in capitals.
    let page = "GARDEN CLUB REGISTRATION\nMEMBER NAME: Robin Example\nPREFERRED NAME: Robin\n\n\
                CLUB INFORMATION\nCLUB NAME: Mossbury Allotment Society - North Field\n\
                CLUB ADDRESS: 3 Orchard Road, Mossbury, MB 40210\n\n\
                PLOT SIZE: Half plot\nSTART OF SEASON: 1 March 2027\nYEARLY FEE: 45 Euro";
    let doc = engine().index_pages(&[(1, page)]).unwrap();
    found(&doc, "Club address?", "3 Orchard Road");
    found(&doc, "What is the club address?", "3 Orchard Road");
    found(&doc, "Yearly fee?", "45 Euro");
    found(&doc, "How much is the yearly fee?", "45 Euro");
    not_found(&doc, "What is the member's email address?");
}

#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn a_form_page_keeps_its_rows_next_to_a_page_of_prose() {
    // R-006: the form's rows on page 1, a page of rules on page 2 (fewer than 40 % rows in the whole document).
    let form = "GARDEN CLUB REGISTRATION\nMEMBER NAME: Robin Example\nCLUB NAME: Mossbury Allotment Society\n\
                CLUB ADDRESS: 3 Orchard Road, Mossbury\nPLOT SIZE: Half plot\nSTART OF SEASON: 1 March 2027\n\
                YEARLY FEE: 45 Euro\nWATER ACCESS: Shared tap";
    let rules = [
        "Members keep their plot tidy and free of weeds throughout the season.",
        "Paths between plots stay clear so that wheelbarrows can pass.",
        "Bonfires are not allowed on any part of the site.",
        "Hosepipes may only be used in the early morning and in the evening.",
        "Sheds must not be taller than two metres.",
        "Members store their tools in their own shed or take them home.",
        "Dogs are welcome on a lead and must not enter other plots.",
        "Children visit the site only with an adult member.",
        "Compost heaps are kept at the back of each plot.",
        "Chemical weed killers are not used anywhere on the site.",
        "The main gate is locked every evening at dusk.",
        "Members tell the committee before they plant fruit trees.",
        "Rainwater butts are encouraged on every shed.",
        "Rubbish that cannot be composted is taken home.",
        "Noise is kept low on Sunday mornings.",
        "Visitors sign the book in the clubhouse.",
        "Members help with the shared hedges twice a year.",
        "A plot left untended for three months may be offered to the next person on the waiting list.",
        "Greenhouses need the committee's written agreement.",
        "Bees may be kept only after the beekeeping course.",
        "The clubhouse kettle is for everyone and is cleaned after use.",
        "Seeds swapped at the spring market are labelled with their variety.",
        "Car parking is limited to the gravel area by the gate.",
        "Ponds are fenced or covered if children visit the plot.",
        "Members report broken taps to the water officer.",
        "The annual meeting takes place in the clubhouse in November.",
        "Prizes for the best plot are given at the summer show.",
        "Wood chips from the tree surgeons are free for all members.",
        "Plots are inspected by the committee in June and September.",
        "Questions about these rules go to the club secretary.",
    ]
    .join(" ");
    let doc = engine().index_pages(&[(1, form), (2, &rules)]).unwrap();
    found(&doc, "Club address?", "3 Orchard Road");
    found(&doc, "Club name?", "Mossbury Allotment Society");
    found(&doc, "Yearly fee?", "45 Euro");
    found(&doc, "Start of season?", "1 March 2027");
    not_found(&doc, "What is the member's email address?");
}

/// A few "label: value" rows inside ordinary text: too few for a form page, still answered when named.
#[test]
#[ignore = "needs the ONNX models and ONNX Runtime (LEAFMIND_QA_MODELS, LEAFMIND_ORT)"]
fn a_few_rows_in_ordinary_text_are_found_by_their_label() {
    let page = "Equipment Loan Notice\n\n\
        Thank you for your request to borrow equipment from the workshop. We are pleased to confirm that it has \
        been approved for the coming term. Please read this notice carefully before you collect anything.\n\n\
        The workshop lends tools to members who have finished the basic safety course. Every item is checked \
        by a technician before it leaves the building. Damage found on return is recorded in the loan book.\n\n\
        Reference ID: LN4471\nDepartment: Woodworking\nTechnician: Sam Rivers\nReturn Date: 30 April\n\n\
        Members should bring their workshop card when they come to collect the equipment. The store is open on \
        weekdays between nine and four. Large items need a trolley, which can be reserved at the front desk.\n\n\
        If the equipment stops working, stop using it at once and tell a member of staff. Do not try to repair \
        anything yourself. A replacement will be offered whenever one is available.";
    let doc = engine().index_pages(&[(1, page)]).unwrap();
    found(&doc, "what is reference ID?", "LN4471");
    found(&doc, "what is the department?", "Woodworking");
    found(&doc, "who is the technician?", "Sam Rivers");
    not_found(&doc, "what is the insurance premium?");
}

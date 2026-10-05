//! The language check: questions must be asked in the document's language (the models match a question to
//! sentences best in the same language). Uses lingua with only the four supported languages.

use lingua::{LanguageDetector, LanguageDetectorBuilder};

/// Languages leafmind-qa supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    German,
    Persian,
    Arabic,
}

const LINGUA: [(lingua::Language, Language); 4] = [
    (lingua::Language::English, Language::English),
    (lingua::Language::German, Language::German),
    (lingua::Language::Persian, Language::Persian),
    (lingua::Language::Arabic, Language::Arabic),
];

/// How much of the document is looked at to find its language, in characters.
const DOCUMENT_SAMPLE: usize = 3000;
/// A question is refused only when the document's own language gets less than this confidence, so short or
/// mixed questions (names, numbers, loan words) still get through, while a question that is clearly in
/// another language is refused even if lingua cannot tell which one (e.g. English or German).
const REFUSE_BELOW: f64 = 0.2;

/// Finds the language of a document and refuses questions clearly asked in another one.
pub struct LanguageCheck(LanguageDetector);

impl LanguageCheck {
    pub fn new() -> Self {
        let languages: Vec<lingua::Language> = LINGUA.iter().map(|(l, _)| *l).collect();
        LanguageCheck(
            LanguageDetectorBuilder::from_languages(&languages)
                .with_preloaded_language_models()
                .build(),
        )
    }

    /// The most likely language of `text` and its confidence (0..1), or `None` for text without letters.
    fn best(&self, text: &str) -> Option<(Language, f64)> {
        let (lang, confidence) = *self.0.compute_language_confidence_values(text).first()?;
        let ours = LINGUA.iter().find(|(l, _)| *l == lang)?.1;
        Some((ours, confidence))
    }

    /// The document's language, from the start of its text; `None` when lingua cannot tell (e.g. only
    /// numbers), and then no question is refused.
    pub fn document(&self, text: &str) -> Option<Language> {
        let sample: String = text.chars().take(DOCUMENT_SAMPLE).collect();
        let lang = self.0.detect_language_of(sample)?;
        Some(LINGUA.iter().find(|(l, _)| *l == lang)?.1)
    }

    /// The question's most likely language if it is clearly not the document's; `None` means the question may
    /// be asked.
    pub fn refuse(&self, document: Language, question: &str) -> Option<Language> {
        let (lang, _) = self.best(question)?;
        let theirs = LINGUA.iter().find(|(_, l)| *l == document)?.0;
        let own = self.0.compute_language_confidence(question, theirs);
        (lang != document && own < REFUSE_BELOW).then_some(lang)
    }
}

impl Default for LanguageCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Language::*;

    #[test]
    fn document_language() {
        let check = LanguageCheck::new();
        assert_eq!(
            check
                .document("Der Mieter zahlt die Miete jeweils bis zum dritten Werktag des Monats."),
            Some(German)
        );
        assert_eq!(
            check.document("اجاره‌بها در ابتدای هر ماه پرداخت می‌شود."),
            Some(Persian)
        );
        assert_eq!(check.document("12 345 — 67"), None);
    }

    #[test]
    fn same_language_questions_pass() {
        let check = LanguageCheck::new();
        assert_eq!(check.refuse(English, "How much is the plot rent?"), None);
        assert_eq!(check.refuse(German, "Wie hoch ist die Kaution?"), None);
        assert_eq!(check.refuse(Arabic, "ما هي مدة العقد؟"), None);
        // Short and mixed questions are not refused.
        assert_eq!(check.refuse(German, "IBAN?"), None);
    }

    #[test]
    fn other_language_questions_are_refused() {
        let check = LanguageCheck::new();
        assert_eq!(
            check.refuse(English, "Wie hoch ist die jährliche Pacht für den Garten?"),
            Some(German)
        );
        assert_eq!(
            check.refuse(German, "How long is the notice period for the tenant?"),
            Some(English)
        );
        assert_eq!(
            check.refuse(Arabic, "مبلغ اجاره ماهانه چقدر است؟"),
            Some(Persian)
        );
        // Clearly not Persian, even though lingua is unsure between English and German.
        assert!(check.refuse(Persian, "How much is the rent?").is_some());
    }
}

//! The rules around the two models, kept apart from them so they can be tested without models: which fields
//! the fine-tuned Needle sees as its tools, how its answer is read, and how the proposal is decided from gte's
//! field ranking, Needle's call and a few word rules (measured as "hybrid4" on public IRS forms, 2026-10-27).

use crate::{Action, Change, Proposal};
use regex::Regex;
use std::sync::LazyLock;

/// Field groups: what a command does to a field of the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Group {
    Text,
    Tick,
    Choice,
}

/// A field that may be meant, with gte's score (best neighbourhood text × its place weight) and that text.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Candidate {
    /// Index into the form's fields.
    pub field: usize,
    pub group: Group,
    pub score: f32,
    pub text: String,
    /// A choice field's options.
    pub options: Vec<String>,
}

/// gte's candidates for Needle: the 5 best text fields, 3 best checkboxes and the best choice field.
pub(crate) fn shortlist(mut all: Vec<Candidate>) -> Vec<Candidate> {
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    let take = |g: Group, n: usize| all.iter().filter(move |c| c.group == g).take(n).cloned();
    take(Group::Text, 5)
        .chain(take(Group::Tick, 3))
        .chain(take(Group::Choice, 1))
        .collect()
}

/// A candidate's name in Needle's tools: its best text, cut to 60 characters as in training.
fn tool_label(c: &Candidate) -> String {
    c.text.chars().take(60).collect()
}

/// Needle's tools for these candidates, written exactly as the training data had them (Python's `json.dumps`).
/// Returns the JSON and, per tool label, the candidate it names (the first one with that label).
pub(crate) fn tools(cands: &[Candidate]) -> (String, Vec<(String, usize)>) {
    let mut labels: Vec<(String, usize)> = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let l = tool_label(c);
        if !labels.iter().any(|(m, _)| *m == l) {
            labels.push((l, i));
        }
    }
    let list = |g: Group| {
        labels
            .iter()
            .filter(|(_, i)| cands[*i].group == g)
            .map(|(l, _)| py_str(l))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut t = vec![format!(
        r#"{{"name": "fill_field", "description": "Write a value into a text field of the form.", "parameters": {{"type": "object", "properties": {{"field": {{"type": "string", "enum": [{}], "description": "The field's label"}}, "value": {{"type": "string", "description": "The value to write, copied from the request"}}}}, "required": ["field", "value"]}}}}"#,
        list(Group::Text)
    )];
    let ticks = list(Group::Tick);
    if !ticks.is_empty() {
        t.push(format!(
            r#"{{"name": "set_checkbox", "description": "Tick or untick a checkbox of the form.", "parameters": {{"type": "object", "properties": {{"field": {{"type": "string", "enum": [{ticks}]}}, "checked": {{"type": "boolean"}}}}, "required": ["field", "checked"]}}}}"#
        ));
    }
    for (l, i) in &labels {
        let c = &cands[*i];
        if c.group == Group::Choice {
            let options: Vec<String> = c.options.iter().map(|o| py_str(o)).collect();
            t.push(format!(
                r#"{{"name": "choose_option", "description": {}, "parameters": {{"type": "object", "properties": {{"option": {{"type": "string", "enum": [{}]}}}}, "required": ["option"]}}}}"#,
                py_str(&format!("Choose one option for '{l}'.")),
                options.join(", ")
            ));
        }
    }
    t.push(r#"{"name": "no_matching_field", "description": "The form has no field for this request, or it is unclear which field is meant.", "parameters": {"type": "object", "properties": {}}}"#.into());
    let json = t
        .iter()
        .map(|f| format!(r#"{{"type": "function", "function": {f}}}"#))
        .collect::<Vec<_>>()
        .join(", ");
    (format!("[{json}]"), labels)
}

/// A JSON string as Python's `json.dumps` writes it (non-ASCII as `\uXXXX`).
fn py_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One of Needle's calls on a candidate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Call {
    Fill { cand: usize, value: String },
    Tick { cand: usize, checked: bool },
    Choose { cand: usize, option: String },
}

/// Needle's calls from its JSON answer (`function_calls`); calls on fields it was not offered are dropped.
pub(crate) fn calls(answer: &str, cands: &[Candidate], labels: &[(String, usize)]) -> Vec<Call> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(answer) else {
        return vec![];
    };
    let by_label = |a: &serde_json::Value| {
        let f = a["field"].as_str()?;
        labels.iter().find(|(l, _)| l == f).map(|(_, i)| *i)
    };
    let mut out = vec![];
    for call in v["function_calls"].as_array().into_iter().flatten() {
        let a = &call["arguments"];
        let c = match call["name"].as_str() {
            Some("fill_field") => by_label(a).map(|cand| Call::Fill {
                cand,
                value: match &a["value"] {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Null => String::new(),
                    other => other.to_string(),
                },
            }),
            Some("set_checkbox") => by_label(a).map(|cand| Call::Tick {
                cand,
                checked: a["checked"].as_bool().unwrap_or(true),
            }),
            Some("choose_option") => {
                let option = a["option"].as_str().unwrap_or_default();
                cands
                    .iter()
                    .position(|c| c.group == Group::Choice && c.options.iter().any(|o| o == option))
                    .map(|cand| Call::Choose {
                        cand,
                        option: option.to_string(),
                    })
            }
            _ => None,
        };
        out.extend(c);
    }
    out
}

/// The value of a fill call that Needle's engine held back (`suppressed_calls`, its own grounding check), for when
/// no other rule finds one: the value is still copied from the command, though the field may be off.
pub(crate) fn held_back_value(answer: &str) -> Option<String> {
    let v = serde_json::from_str::<serde_json::Value>(answer).ok()?;
    v["suppressed_calls"]
        .as_array()?
        .iter()
        .filter(|c| c["name"] == "fill_field")
        .find_map(|c| c["arguments"]["value"].as_str().map(str::to_string))
}

static TICK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(tick|untick|check|uncheck|select|mark)\b").unwrap());
static UNTICK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(untick|uncheck|unmark|deselect)\b").unwrap());
static VALUE_LIKE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d|@|:\s*\S").unwrap());
static AFTER_IS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:is|are|was)\s+(.+)$").unwrap());
static FROM_DIGIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\S*\d.*)$").unwrap());
static TODAY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\btoday\b").unwrap());
static SIGN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(please\s+)?(sign\b|add\s+(my\s+|a\s+)?signature\b)").unwrap()
});
static CLEAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(please\s+)?(clear|empty|erase|delete|remove)\b").unwrap()
});
static SAME_AS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(.*?)\bsame\s+as\b(.*)$").unwrap());

/// "sign …": signatures are not written as text.
pub(crate) fn is_signing(command: &str) -> bool {
    SIGN.is_match(command)
}

/// "clear …", "empty …", "remove …".
pub(crate) fn is_clearing(command: &str) -> bool {
    CLEAR.is_match(command)
}

/// "X same as Y" → (X, Y), both non-empty.
pub(crate) fn same_as(command: &str) -> Option<(&str, &str)> {
    let m = SAME_AS.captures(command)?;
    let (a, b) = (
        m.get(1)?
            .as_str()
            .trim_matches(|c: char| !c.is_alphanumeric()),
        m.get(2)?
            .as_str()
            .trim_matches(|c: char| !c.is_alphanumeric()),
    );
    (!a.is_empty() && !b.is_empty()).then_some((a, b))
}

/// The value to write for a fill command: Needle's copy; else the text after the last ":"; else after
/// "is/are/was"; else a value Needle held back; else the command's tail from its first token with a digit.
/// "today" becomes `today`.
fn value(
    command: &str,
    needle: Option<&str>,
    held_back: Option<&str>,
    today: &str,
) -> Option<String> {
    let v = needle
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or_else(|| command.rsplit_once(':').map(|(_, v)| v.trim().to_string()))
        .or_else(|| AFTER_IS.captures(command).map(|m| m[1].trim().to_string()))
        .or_else(|| {
            held_back
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
        .or_else(|| FROM_DIGIT.captures(command).map(|m| m[1].to_string()))
        .filter(|v| !v.is_empty());
    let says_today = |v: &str| {
        let v = v
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        matches!(
            v.as_str(),
            "today" | "today's date" | "todays date" | "the date of today"
        )
    };
    match v {
        Some(v) if says_today(&v) => Some(today.to_string()),
        None if TODAY.is_match(command) => Some(today.to_string()),
        v => v,
    }
}

/// The best candidate of `group` with the next two as alternatives (candidates sorted best first).
fn best_of(cands: &[Candidate], group: Group) -> Option<(usize, Vec<usize>)> {
    let mut of = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.group == group)
        .map(|(i, _)| i);
    Some((of.next()?, of.take(2).collect()))
}

/// The proposal for `command` from gte's shortlist (sorted best first within each group) and Needle's calls.
/// Field indices in the result are the candidates' `field`s; the engine maps them to the app's ids.
pub(crate) fn decide(
    command: &str,
    today: &str,
    cands: &[Candidate],
    calls: &[Call],
    held_back: Option<&str>,
) -> Proposal {
    let action = |cand: usize, change: Change, alternatives: Vec<usize>| Action {
        field: cands[cand].field as u32,
        change,
        alternatives: alternatives
            .into_iter()
            .map(|i| cands[i].field as u32)
            .collect(),
    };
    let others = |cand: usize| {
        let g = cands[cand].group;
        (0..cands.len())
            .filter(|&i| i != cand && cands[i].group == g)
            .take(2)
            .collect::<Vec<_>>()
    };
    // Several fields at once ("name Robin Example, phone 555 0100"): Needle's own fields.
    let mut seen = vec![];
    let multi: Vec<&Call> = calls
        .iter()
        .filter(|c| match c {
            Call::Fill { cand, value } => {
                !value.trim().is_empty() && !seen.contains(cand) && {
                    seen.push(*cand);
                    true
                }
            }
            Call::Tick { cand, .. } => {
                !seen.contains(cand) && {
                    seen.push(*cand);
                    true
                }
            }
            Call::Choose { .. } => false,
        })
        .collect();
    if multi.len() >= 2 {
        return Proposal::Actions(
            multi
                .into_iter()
                .map(|c| match c {
                    Call::Fill { cand, value } => {
                        action(*cand, Change::Fill(value.trim().to_string()), others(*cand))
                    }
                    Call::Tick { cand, checked } => {
                        action(*cand, Change::Check(*checked), others(*cand))
                    }
                    Call::Choose { .. } => unreachable!(),
                })
                .collect(),
        );
    }
    // A choice field: Needle chose an option, or the command names one of the best choice field's options.
    if let Some(Call::Choose { cand, option }) = calls.first() {
        return Proposal::Actions(vec![action(*cand, Change::Choose(option.clone()), vec![])]);
    }
    let lower = command.to_lowercase();
    if let Some(c) = cands.iter().position(|c| c.group == Group::Choice)
        && cands
            .iter()
            .all(|o| o.group == Group::Choice || o.score <= cands[c].score)
        && let Some(o) = cands[c].options.iter().find(|o| {
            Regex::new(&format!(r"\b{}\b", regex::escape(&o.to_lowercase())))
                .is_ok_and(|r| r.is_match(&lower))
        })
    {
        return Proposal::Actions(vec![action(c, Change::Choose(o.clone()), vec![])]);
    }
    // Fill or tick.
    let tick = if TICK.is_match(command) {
        true
    } else if let Some(c) = calls.first() {
        matches!(c, Call::Tick { .. })
    } else if VALUE_LIKE.is_match(command) {
        false
    } else {
        match cands
            .iter()
            .filter(|c| c.group != Group::Choice)
            .max_by(|a, b| a.score.total_cmp(&b.score))
        {
            Some(c) => c.group == Group::Tick,
            None => return Proposal::NoSuchField,
        }
    };
    let Some((best, alternatives)) = best_of(cands, if tick { Group::Tick } else { Group::Text })
    else {
        return Proposal::NoSuchField;
    };
    if tick {
        let checked = match calls.first() {
            Some(Call::Tick { checked, .. }) => *checked,
            _ => !UNTICK.is_match(command),
        };
        return Proposal::Actions(vec![action(best, Change::Check(checked), alternatives)]);
    }
    let needle = match calls.first() {
        Some(Call::Fill { value, .. }) => Some(value.as_str()),
        _ => None,
    };
    match value(command, needle, held_back, today) {
        Some(v) => Proposal::Actions(vec![action(best, Change::Fill(v), alternatives)]),
        None => Proposal::NoValue {
            candidates: std::iter::once(best)
                .chain(alternatives)
                .map(|i| cands[i].field as u32)
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(field: usize, group: Group, score: f32, text: &str) -> Candidate {
        Candidate {
            field,
            group,
            score,
            text: text.into(),
            options: vec![],
        }
    }

    fn form() -> Vec<Candidate> {
        shortlist(vec![
            cand(0, Group::Text, 0.8, "Full name"),
            cand(1, Group::Text, 0.6, "Phone"),
            cand(2, Group::Text, 0.5, "Email"),
            cand(3, Group::Tick, 0.4, "I agree to the terms"),
            cand(4, Group::Text, 0.3, "City"),
        ])
    }

    fn one(p: Proposal) -> Action {
        match p {
            Proposal::Actions(mut a) if a.len() == 1 => a.remove(0),
            other => panic!("expected one action, got {other:?}"),
        }
    }

    #[test]
    fn tools_as_python_writes_them() {
        let (json, labels) = tools(&form());
        assert!(json.starts_with(r#"[{"type": "function", "function": {"name": "fill_field", "description": "Write a value into a text field of the form.", "parameters": {"type": "object", "properties": {"field": {"type": "string", "enum": ["Full name", "Phone", "Email", "City"]"#));
        assert!(
            json.contains(r#""enum": ["I agree to the terms"]}, "checked": {"type": "boolean"}}"#)
        );
        assert!(json.ends_with(r#""properties": {}}}}]"#));
        assert_eq!(labels.len(), 5);
        assert_eq!(
            py_str("Designee’s \"name\""),
            r#""Designee\u2019s \"name\"""#
        );
        assert_eq!(py_str("€𝄞"), r#""\u20ac\ud834\udd1e""#);
        serde_json::from_str::<serde_json::Value>(&json).unwrap();
    }

    #[test]
    fn needle_calls_are_read() {
        let c = form();
        let (_, labels) = tools(&c);
        let answer = r#"{"type":"call","function_calls":[{"name":"fill_field","arguments":{"field":"Phone","value":"555 0100"}},{"name":"fill_field","arguments":{"field":"Not offered","value":"x"}},{"name":"set_checkbox","arguments":{"field":"I agree to the terms","checked":false}}]}"#;
        assert_eq!(
            calls(answer, &c, &labels),
            vec![
                Call::Fill {
                    cand: 1,
                    value: "555 0100".into()
                },
                Call::Tick {
                    cand: 4,
                    checked: false
                }
            ]
        );
        assert!(calls("not json", &c, &labels).is_empty());
        let held = r#"{"function_calls":[],"suppressed_calls":[{"name":"fill_field","arguments":{"field":"Phone","value":"Green Thumb"}}]}"#;
        assert_eq!(held_back_value(held).as_deref(), Some("Green Thumb"));
        let p = decide("trade name Green Thumb", "", &c, &[], Some("Green Thumb"));
        assert_eq!(one(p).change, Change::Fill("Green Thumb".into()));
    }

    #[test]
    fn hybrid_rules() {
        let c = form();
        // gte's best text field wins over Needle's field; Needle's value is used.
        let a = one(decide(
            "my name is Robin Example",
            "10/27/2026",
            &c,
            &[Call::Fill {
                cand: 1,
                value: "Robin Example".into(),
            }],
            None,
        ));
        assert_eq!(
            (a.field, a.change, a.alternatives),
            (0, Change::Fill("Robin Example".into()), vec![1, 2])
        );
        // Without Needle: the text after ":", after "is", or from the first digit.
        let v = |cmd: &str| match one(decide(cmd, "10/27/2026", &c, &[], None)).change {
            Change::Fill(v) => v,
            other => panic!("{other:?}"),
        };
        assert_eq!(v("name: Robin Example"), "Robin Example");
        assert_eq!(v("my phone number is 555 0100"), "555 0100");
        assert_eq!(v("phone 555 0100"), "555 0100");
        assert_eq!(v("the date is today"), "10/27/2026");
        // A tick word makes it a tick, on the best checkbox; "untick" unticks.
        assert_eq!(
            one(decide("tick I agree", "", &c, &[], None)).change,
            Change::Check(true)
        );
        assert_eq!(one(decide("untick the terms", "", &c, &[], None)).field, 3);
        assert_eq!(
            one(decide("untick the terms", "", &c, &[], None)).change,
            Change::Check(false)
        );
        // No value anywhere: ask for it.
        assert_eq!(
            decide("fill in my name", "", &c, &[], None),
            Proposal::NoValue {
                candidates: vec![0, 1, 2]
            }
        );
        // Tick but no checkbox on the form.
        let texts: Vec<Candidate> = c
            .iter()
            .filter(|c| c.group == Group::Text)
            .cloned()
            .collect();
        assert_eq!(
            decide("tick the box", "", &texts, &[], None),
            Proposal::NoSuchField
        );
        // Two of Needle's fills: both, on Needle's fields.
        let p = decide(
            "name Robin, phone 555",
            "",
            &c,
            &[
                Call::Fill {
                    cand: 0,
                    value: "Robin".into(),
                },
                Call::Fill {
                    cand: 1,
                    value: "555".into(),
                },
            ],
            None,
        );
        assert!(matches!(&p, Proposal::Actions(a) if a.len() == 2 && a[1].field == 1));
    }

    #[test]
    fn word_rules() {
        assert!(is_signing("sign the form") && is_signing("please add my signature"));
        assert!(!is_signing("signature date: 10/27/2026"));
        assert!(is_clearing("clear the phone field") && !is_clearing("my phone is 555"));
        assert_eq!(
            same_as("mailing address same as home address"),
            Some(("mailing address", "home address"))
        );
        let mut c = form();
        c.push(Candidate {
            options: vec!["Married".into(), "Single".into()],
            ..cand(5, Group::Choice, 0.9, "Marital status")
        });
        assert_eq!(
            one(decide("I am married", "", &c, &[], None)).change,
            Change::Choose("Married".into())
        );
    }
}

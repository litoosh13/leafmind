//! The steps around the two models, shared by the native engine and the browser build: what each field is
//! compared by, the ranking, and the proposal from the ranking and Needle's answer. The engines only run gte and
//! Needle between these steps.

use crate::decide::{self, Candidate, Group};
use crate::{Action, Change, FieldKind, FormField, PageWords, Proposal};
use leafmind_fields::{LabelFor, field_context, tooltip_label};
use std::collections::HashMap;

/// Weight of a text by where it stands from a text field (or dropdown) and from a checkbox: a checkbox's label
/// is usually on its right. Fixed before the field-matching test on public forms (2026-10-23).
fn weight(group: Group, place: &str) -> f32 {
    match (group, place) {
        (Group::Tick, "right") => 1.0,
        (Group::Tick, "left") => 0.95,
        (Group::Tick, "above" | "inside") => 0.9,
        (Group::Tick, "below") => 0.85,
        (_, "below") => 0.9,
        (_, "right") => 0.85,
        _ => 1.0,
    }
}

/// A form prepared for commands: each field's surrounding texts, embedded once.
pub struct Form {
    fields: Vec<Prepared>,
    vectors: Vec<Vec<f32>>,
}

pub(crate) struct Prepared {
    id: u32,
    group: Group,
    options: Vec<String>,
    read_only: bool,
    /// (weight, text, index into the distinct texts).
    texts: Vec<(f32, String, usize)>,
}

/// Each field's surrounding texts (from its page's words, and a usable tooltip) and the distinct texts to embed.
pub(crate) fn layout(fields: &[FormField], pages: &[PageWords]) -> (Vec<Prepared>, Vec<String>) {
    let mut prepared: Vec<Prepared> = fields
        .iter()
        .map(|f| Prepared {
            id: f.id,
            group: match f.kind {
                FieldKind::Text => Group::Text,
                FieldKind::Checkbox => Group::Tick,
                FieldKind::Choice { .. } => Group::Choice,
            },
            options: match &f.kind {
                FieldKind::Choice { options } => options.clone(),
                _ => vec![],
            },
            read_only: f.read_only,
            texts: vec![],
        })
        .collect();
    let mut distinct: Vec<String> = vec![];
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut index = |t: String| -> (String, usize) {
        let i = *seen.entry(t.clone()).or_insert_with(|| {
            distinct.push(t.clone());
            distinct.len() - 1
        });
        (t, i)
    };
    let empty = vec![];
    let mut numbers: Vec<u32> = fields.iter().map(|f| f.page).collect();
    numbers.sort_unstable();
    numbers.dedup();
    for page in numbers {
        let on: Vec<usize> = (0..fields.len())
            .filter(|&i| fields[i].page == page)
            .collect();
        let boxes: Vec<([f32; 4], LabelFor)> = on
            .iter()
            .map(|&i| {
                let label = match fields[i].kind {
                    FieldKind::Checkbox => LabelFor::Choice,
                    _ => LabelFor::Text,
                };
                (fields[i].bounds, label)
            })
            .collect();
        let words = pages
            .iter()
            .find(|p| p.page == page)
            .map_or(&empty, |p| &p.words);
        for (&i, ctx) in on.iter().zip(field_context(&boxes, words)) {
            if fields[i].read_only {
                continue;
            }
            let g = prepared[i].group;
            for (place, t) in ctx {
                let (t, v) = index(t);
                prepared[i].texts.push((weight(g, place), t, v));
            }
            if let Some(t) = fields[i].tooltip.as_deref().and_then(tooltip_label) {
                let (t, v) = index(t);
                prepared[i].texts.push((1.0, t, v));
            }
        }
    }
    (prepared, distinct)
}

impl Form {
    /// From [`layout`] and the embeddings of its distinct texts (same order).
    pub(crate) fn new(fields: Vec<Prepared>, vectors: Vec<Vec<f32>>) -> Self {
        Form { fields, vectors }
    }
}

/// Every field that can be proposed, scored against a text's embedding `q` (fields without any text around are
/// left out).
pub(crate) fn rank(form: &Form, q: &[f32]) -> Vec<Candidate> {
    let dot = |v: &[f32]| q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
    form.fields
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.read_only)
        .filter_map(|(i, f)| {
            let (score, text) = f
                .texts
                .iter()
                .map(|(w, t, v)| (dot(&form.vectors[*v]) * w, t))
                .max_by(|a, b| a.0.total_cmp(&b.0))?;
            Some(Candidate {
                field: i,
                group: f.group,
                score,
                text: text.clone(),
                options: f.options.clone(),
            })
        })
        .collect()
}

/// What to do first with a command.
pub(crate) enum Plan<'c> {
    Done(Proposal),
    /// Rank the fields against both parts, then [`copy_from`].
    SameAs {
        target: &'c str,
        source: &'c str,
    },
    /// Rank the fields against the command, then [`after_rank`].
    Rank,
}

pub(crate) fn plan(command: &str) -> Plan<'_> {
    if decide::is_signing(command) {
        return Plan::Done(Proposal::NotSupported {
            reason: "signatures are added with a signing tool, not typed",
        });
    }
    match decide::same_as(command) {
        Some((target, source)) => Plan::SameAs { target, source },
        None => Plan::Rank,
    }
}

/// "X same as Y": the best text field for X gets the value of the best other text field for Y.
pub(crate) fn copy_from(form: &Form, to: Vec<Candidate>, from: Vec<Candidate>) -> Proposal {
    let texts = |c: Vec<Candidate>| -> Vec<Candidate> {
        decide::shortlist(c)
            .into_iter()
            .filter(|c| c.group == Group::Text)
            .collect()
    };
    let id = |c: &Candidate| form.fields[c.field].id;
    let to = texts(to);
    let Some(t) = to.first() else {
        return Proposal::NoSuchField;
    };
    let Some(s) = texts(from).into_iter().find(|s| s.field != t.field) else {
        return Proposal::NoSuchField;
    };
    Proposal::Actions(vec![Action {
        field: id(t),
        change: Change::CopyFrom(id(&s)),
        alternatives: to[1..].iter().take(2).map(id).collect(),
    }])
}

/// Needle's question: its tools for the shortlisted candidates.
pub(crate) struct Ask {
    pub tools: String,
    labels: Vec<(String, usize)>,
    cands: Vec<Candidate>,
}

pub(crate) enum Next {
    Done(Proposal),
    /// Run Needle on the command with `tools` (and [`system`]), then [`finish`].
    Ask(Ask),
}

/// After ranking the fields against the command: clearing, ticking and "label: value" are decided here (see
/// `decide::rules_decide`), everything else needs Needle.
pub(crate) fn after_rank(form: &Form, command: &str, today: &str, all: Vec<Candidate>) -> Next {
    let cands = decide::shortlist(all);
    if decide::is_clearing(command) {
        let id = |c: &Candidate| form.fields[c.field].id;
        let Some(best) = cands
            .iter()
            .filter(|c| c.group != Group::Choice)
            .max_by(|a, b| a.score.total_cmp(&b.score))
        else {
            return Next::Done(Proposal::NoSuchField);
        };
        let same: Vec<&Candidate> = cands.iter().filter(|c| c.group == best.group).collect();
        return Next::Done(Proposal::Actions(vec![Action {
            field: id(best),
            change: if best.group == Group::Tick {
                Change::Check(false)
            } else {
                Change::Clear
            },
            alternatives: same[1..].iter().take(2).map(|c| id(c)).collect(),
        }]));
    }
    let (tools, labels) = decide::tools(&cands);
    let ask = Ask {
        tools,
        labels,
        cands,
    };
    if decide::rules_decide(command) {
        return Next::Done(finish(form, command, today, &ask, ""));
    }
    Next::Ask(ask)
}

/// The proposal from Needle's JSON `answer`, with the app's field ids.
pub(crate) fn finish(form: &Form, command: &str, today: &str, ask: &Ask, answer: &str) -> Proposal {
    let calls = decide::calls(answer, &ask.cands, &ask.labels);
    let held_back = decide::held_back_value(answer);
    let id = |f: u32| form.fields[f as usize].id;
    match decide::decide(command, today, &ask.cands, &calls, held_back.as_deref()) {
        Proposal::Actions(actions) => Proposal::Actions(
            actions
                .into_iter()
                .map(|a| Action {
                    field: id(a.field),
                    change: match a.change {
                        Change::CopyFrom(f) => Change::CopyFrom(id(f)),
                        c => c,
                    },
                    alternatives: a.alternatives.into_iter().map(id).collect(),
                })
                .collect(),
        ),
        Proposal::NoValue { candidates } => Proposal::NoValue {
            candidates: candidates.into_iter().map(id).collect(),
        },
        p => p,
    }
}

/// Needle's system prompt: the date line its Python library adds, as in training ("date: 2026-10-27 Tue 10:00",
/// UTC), from seconds since 1970.
pub(crate) fn system(secs: i64) -> String {
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let weekday = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
    format!(
        "date: {y}-{m:02}-{d:02} {weekday} {:02}:{:02}",
        rest / 3600,
        rest / 60 % 60
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn system_date_line() {
        assert_eq!(super::system(0), "date: 1970-01-01 Thu 00:00");
        assert_eq!(super::system(1_793_095_200), "date: 2026-10-27 Tue 10:00");
        assert_eq!(super::system(951_782_400), "date: 2000-02-29 Tue 00:00");
    }
}

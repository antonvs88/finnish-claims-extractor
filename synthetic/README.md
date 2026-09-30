# Synthetic boolean question–answer set

A fixed test set for the 24 boolean fields: a Finnish question followed by an explicit yes, no or unknown answer.
It checks one thing, whether the extractor reads explicit answers in wordings it has not been tuned on. It says
nothing about real claims. The wordings were written by Claude, not by a native annotator: have a Finnish speaker
read `phrasings.py` before the set is used to judge a model.

| split | cases | what it holds |
|---|---:|---|
| dev | 432 | question wordings 0–2, wrappers 0–1, answer wordings 0–1 |
| holdout | 144 | question wording 3, answer wording 2, wrappers 1–2: none of the three seen in dev |
| smoke | 72 | the first dev case of every field and class; the quick check for a pull request (about 5 min on 8 CPU cores) |

`python synthetic/build.py` regenerates `cases.json` and `input-*.json` (deterministic).
Run the extractor on an input file, then score it:

    python run.py --input synthetic/input-smoke.json --output pred.json
    python synthetic/score.py pred.json --split smoke --baseline synthetic/baseline-smoke.json

`score.py` prints per-field counts only and exits 1 when the run is worse than the baseline: a field with fewer
correct answers, more wrong answers, or more spurious answers on other boolean fields. "Missed" means null where the
text gave an answer (safe for approvals, costs coverage); "wrong" means a swapped or invented answer.

Baselines (30.9.2026, frozen candidate, CPU float32): smoke 62/72 correct, holdout 118/144, dev 364/432. Wrong answers
are rare (2 in dev, 0 in holdout); nearly all errors are nulls, concentrated in `damageCausedByListedCyberEvent`,
`isDirectConsequenceOfCoveredEvent`, `isLicenseOrSkillCourseTraining`, `nuclearDamagePresent` and
`ylapuolisenEsteenKuljetuspoikkeus`.

Rules for anyone, human or agent, improving the model against this set: tune on dev only; score holdout once per
attempt series and read only the aggregate; report the dev–holdout gap every time. The repo's first candidate scored 95%
on its development cases and 16.7% recall on its frozen question–answer holdout, so an improvement on dev alone counts
for nothing. Narrative texts, amounts, dates and enum fields are not covered yet.

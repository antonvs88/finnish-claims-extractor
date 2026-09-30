# Overnight frozen holdout results

**The 95% development result did not generalize. The locked candidate is not demonstrated to improve overall held-out extraction and should not be promoted as a 95%-accurate model.**

No training, threshold adjustment or model selection followed this evaluation. Report the narrative and questionnaire strata separately: same-author synthetic data cannot establish production accuracy.

| Stratum | Model | Known facts correct | Supplied answers correct | Unsupported | Entire claims correct |
|---|---|---:|---:|---:|---:|
| overall | starting_baseline | 62/95 (65.3%) | 62/84 (73.8%) | 21 | 41/90 |
| overall | locked_candidate | 47/95 (49.5%) | 47/63 (74.6%) | 15 | 40/90 |
| narrative | starting_baseline | 39/47 (83.0%) | 39/46 (84.8%) | 6 | 8/18 |
| narrative | locked_candidate | 39/47 (83.0%) | 39/45 (86.7%) | 5 | 9/18 |
| diagnostic_questionnaire | starting_baseline | 23/48 (47.9%) | 23/38 (60.5%) | 15 | 33/72 |
| diagnostic_questionnaire | locked_candidate | 8/48 (16.7%) | 8/18 (44.4%) | 10 | 31/72 |

Matched warmed local throughput by stratum (two repeats):

| Model | Stratum | Tokens | Claims/hour |
|---|---|---:|---:|
| starting_baseline | narrative | 45–80 | 10652 |
| starting_baseline | narrative | 45–80 | 10522 |
| starting_baseline | diagnostic_questionnaire | 70–117 | 6902 |
| starting_baseline | diagnostic_questionnaire | 70–117 | 6895 |
| locked_candidate | narrative | 45–80 | 7629 |
| locked_candidate | narrative | 45–80 | 7506 |
| locked_candidate | diagnostic_questionnaire | 70–117 | 3635 |
| locked_candidate | diagnostic_questionnaire | 70–117 | 2464 |

Per-field scores and Boolean true/false/null confusion are retained in `heldout-metrics.json`. Document-bootstrap intervals are descriptive and available in `uncertainty.json`. Rare field and false-value counts may be too small for useful conclusions.

Development results were 95.3% known-fact recall and 95.7% supplied-answer precision on122 repeatedly used synthetic development cases. The final holdout was frozen before overnight training; normalized exact-overlap audits cannot rule out shared author/style/schema bias. Real anonymized claims and independent annotation are still required.

## Failure analysis (reporting only)

On the 18 narrative claims, the candidate recovered the same 39/47 known facts as the starting baseline. It gained one correct enum value but lost one numeric value. Precision improved by one fewer unsupported answer (6 to 5), and entirely correct narratives improved from 8/18 to 9/18. This is a small, uncertain improvement, not the development gain.

The 72 questionnaire diagnostics expose a large Boolean format-generalization failure. Each asks a Finnish question followed by an explicit verified yes/no/unknown answer. Of 24 true answers, the baseline extracted 22 and the locked candidate only 7; the remaining 17 became null. Both recovered only 1/24 false answers. The candidate therefore improved apparent caution at the expense of useful known-fact recall on this unseen format. The pooled null accuracy is dominated by the other unspecified fields and is not evidence of successful extraction.

This audit identifies the failed behavior, not whether training weights, ensemble blending or the neutral prior caused it individually. No post-holdout calibration, model replacement, training or selection was performed. The locked artifacts remain unchanged. The next independently authorized iteration needs training and development coverage of question–answer entailment and negation, plus a genuinely fresh final holdout; these examined cases can serve only as regressions. Independent Finnish annotation and real anonymized claim samples are still needed.

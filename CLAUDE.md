# finnish-claims-extractor — context for Claude sessions

Research prototype: Finnish ModernBERT plus mDeBERTa NLI extraction of 42 kolarivakuutus fields (24 booleans,
6 enums, 10 amounts, 2 dates), no LLM. It is a candidate replacement for the Sonnet extraction in Fennia's
claims-automation backtest (repo FenniaCommon/business-logic-engine, `fin-anon/backtesting/mapping/local_extractor.py`
calls this checkout). The goal there is to find claim subsets that can be automated; first deployment automates
approvals only, so a wrong answer costs more than a null. Not production-ready: the first candidate scored 95% on
development cases and 16.7% recall on its frozen question–answer holdout.

## Rules
- Only synthetic data here. Never put real claim text, claim numbers or per-claim outputs from Fennia's laptop into
  this repo, its issues, PRs or notes. From the laptop backtest only aggregates come back (counts by field and
  outcome); turn them into synthetic cases, never into copies of claims.
- Keep the repo private. Weights are release assets (SHA-256 in `model-manifest.json`), never Git objects.
- Change through a branch and PR; Anton merges. Do not touch `synthetic/baseline-*.json` except to record a run that
  was accepted on purpose.
- Everything in English.

## Checks
- Environment: `.venv` (Python 3.13, CPU torch 2.14.0). Device is picked automatically (mps, cuda, cpu;
  `CLAIMS_EXTRACTOR_DEVICE` overrides).
- Quick check, about 5 minutes on 8 CPU cores: `python run.py --input synthetic/input-smoke.json --output pred.json`
  then `python synthetic/score.py pred.json --split smoke --baseline synthetic/baseline-smoke.json`
  (exit 1 means a regression). Same for `dev` (about 30 minutes) and `holdout` (about 10 minutes).
- Baselines of the frozen candidate: smoke 62/72, holdout 118/144, dev 364/432 correct; almost every error is a
  null, not a wrong answer.

## Improving the model (agent loops)
- Tune on the dev split only. Score holdout once per attempt series and read only the aggregate. Report the
  dev–holdout gap every time; an improvement on dev alone counts for nothing.
- A loop needs a trigger, a success criterion and a stop condition before it starts (see `docs/feedback-loops.md` in
  the business-logic-engine repo): a fixed number of attempts, or no improvement over two. Promoting a model is
  Anton's decision, never the loop's.
- Keep loop state in files (a log per attempt), not in the conversation.
- The synthetic wordings were written by Claude, not a native Finnish annotator; treat scores as a regression signal,
  not as accuracy on real claims. Narratives, amounts, dates and enums are not covered yet.

## Environment on the VM
- The checkout is `~/finnish-claims-extractor`; the venv is `.venv`. Weights are in `models/` (5 GB, ignored by Git).
- `gh` has two accounts and the active one (Fennia) cannot see this private repo. Use the personal account for it:
  `export GH_TOKEN=$(gh auth token --user antonvs88)`, and for a push
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push -u origin BRANCH`.
- The SessionStart hook may list unacknowledged backtest inbox reports. They belong to the pilot session in
  `~/business-logic-engine`; ignore them here, do not read or acknowledge them, and never arm the inbox watch.
- Fennia's laptop runs this extractor through `mapping/local_extractor.py` in the business-logic-engine repo; a
  change to `run.py`, `config.json` or `model-manifest.json` changes the cache fingerprint there.
- A new session does not inherit earlier conversations. Read this file, `synthetic/README.md` and
  `docs/feedback-loops.md` (business-logic-engine repo) first.

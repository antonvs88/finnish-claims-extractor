# Finnish claims extractor — research prototype

**Not production-ready. The 95% development result did not generalize to the frozen holdout.** This is a private research snapshot for Finnish kolarivakuutus text extraction, not a coverage or payout decision engine.

## Models

- `TurkuNLP/finnish-modernbert-base`, revision `a80cc04b3812deb6eaf6006d23569236ca3ab32b`: fine-tuned numeric/date role extraction with a local-context residual head.
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`: fine-tuned NLI components for booleans, event enums and other enums; a two-model Boolean ensemble, country hypotheses, and development-selected calibration.
- 42 nullable fields: 24 booleans, 6 enums, 10 amounts and 2 dates. No Gemma model is used.

## Results

| Frozen holdout | Starting recall / precision | This candidate |
|---|---:|---:|
| 18 synthetic narratives | 83.0% / 84.8% | 83.0% / 86.7% |
| 72 question–answer diagnostics | 47.9% / 60.5% | 16.7% / 44.4% |

The new Boolean model frequently returns null for explicit yes/no answers in unseen question–answer formats. Development recall/precision of 95.3%/95.7% must not be represented as unseen accuracy. See [full results](reports/heldout-results.md). These are small, same-author synthetic samples; no production accuracy claim is justified.

## Run locally

The device is picked automatically: `mps` (Apple Silicon), else `cuda`, else `cpu`; set `CLAIMS_EXTRACTOR_DEVICE` to force one. The frozen runtime was built and tested on an M3 Max with 128GB RAM (half precision on mps). CPU runs in float32 and reproduced the 2 bundled synthetic outputs and 72 synthetic question–answer outputs identical to half precision on a Linux CPU (72 short texts in under 5 minutes on 8 cores); cuda has not been tested. Full-dataset parity with the mps runtime has not been re-measured off Apple Silicon.

1. Create a Python 3.13 environment and install `requirements.txt`. The recorded Torch build may require the matching prerelease wheel source; use an available compatible build if necessary and recheck parity.
2. Authenticate the GitHub CLI with access to this private repository, then run `python download_models.py`. This verifies and extracts release assets.
3. Run `python verify_models.py`.
4. Run `python run.py --input examples/synthetic.json --output extraction.json`.

Input is a JSON list of objects containing `text` and an optional `id`. Output contains the 42-field prediction. Inference is offline once dependencies and weights are present. Inputs over 4096 ModernBERT tokens fail instead of truncating silently.

Weights are release assets, not Git objects. `model-manifest.json` records their SHA-256 hashes. The packaged initializer omits superseded model loads, retains the frozen inference mathematics, and matches the original runtime on all 5,124 fields across122 development cases, plus the included synthetic smoke cases. This is a selected inference snapshot, not the complete experiment or training reproduction archive.

## Provenance and limitations

Selection was locked before the final holdout evaluation; no tuning followed its results. Prior tests are regressions. Original checkpoints and experiment evidence remain in the local research workspace. The original private business-rules repository, real claims, training datasets, and held-out case texts are not included here.

Domain field definitions/hypotheses are retained because the extractor needs them. Keep this repository private. Upstream models and software retain their own licenses; no new blanket license is granted over third-party or domain-derived material. See `MODEL_SOURCES.md`.

# Implementation validation — 2026-11-13

- Four pytest checks passed: deterministic deduplication/splitting, conflicting labels, RGB preprocessing, and corrupt-image handling.
- Audited all 1,600 repository images: 714 train / 80 validation / 400 test after 406 duplicate exclusions (402 cross-source-split matches).
- A one-epoch CPU training run on the cleaned split completed with TensorFlow 2.16.2.
- The saved `best.keras` checkpoint reloaded for evaluation and single-image CLI prediction.
- History, metadata, exact split manifest, per-class report, confusion matrix and test probabilities were generated.
- The smoke checkpoint is not a trained release and its accuracy is not reported as a benchmark. Run artifacts are excluded from git.
- Full baseline training, multi-seed evaluation, near-duplicate investigation, and a validated public demo remain future phases.

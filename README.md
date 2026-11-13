# Coffee Bean Roast Classifier

An educational computer vision project by **Vijayendra Dwari**: classify coffee-bean photographs as **Dark, Green (unroasted), Light, or Medium** using a small CNN trained from scratch.

## Project links

| Resource | Purpose |
| --- | --- |
| [Medium article](https://medium.com/codex/learning-image-classification-with-cnn-using-tensorflow-4875fdd3ce62) | Original educational walkthrough (2023) |
| [Kaggle notebook](https://www.kaggle.com/code/vijayendrad/coffee-bean-classification-99-cnn) | Historical hosted experiment |
| [Kaggle dataset](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) | Resized dataset distribution |
| [Coffee Roast Intelligence](https://arxiv.org/abs/2206.01841) | Source research by Ontoum et al. (2022) |
| [Original notebook](training_notebook.ipynb) | Preserved repository experiment and outputs |
| [Data audit](reports/data-audit.json) | Reproducible split manifest and duplicate exclusions |
| [Method and limitations](docs/method.md) | Historical differences and evaluation boundaries |

## Important evaluation finding

The repository contains **1,600 images**: 1,200 in `data/train` and 400 in `data/test`, with four equally represented labels. The new audit detects **402 training files matching test images** and **4 additional duplicate training files**, using decoded native-resolution RGB pixel hashes. Historical accuracy is therefore not evidence of performance on an independent test set.

The default pipeline preserves the test set, excludes duplicate training copies, and creates a seeded, stratified validation split from the remaining unique training pool:

| Split | Images |
| --- | ---: |
| Training | 714 |
| Validation | 80 |
| Test | 400 |

Source files are unchanged. Exact deduplication does not eliminate near-duplicate images or prove independence across photography sessions. No new benchmark accuracy is claimed yet.

## Quick start

Use Python 3.10–3.12 (the local smoke test used TensorFlow 2.16.2). From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m coffee_classifier audit
python -m coffee_classifier train --output runs/baseline
```

The included data is used by default. Pass `--data /path/to/data` for another directory containing `train/{Dark,Green,Light,Medium}` and `test/{Dark,Green,Light,Medium}`. Dataset licensing and attribution are separate from the repository's MIT code license; consult the upstream dataset terms before redistribution.

For a one-epoch installation check (not a benchmark):

```bash
python -m coffee_classifier train --epochs 1 --output runs/smoke
```

Predict an image after training:

```bash
python -m coffee_classifier predict --run runs/baseline --image path/to/beans.jpg
```

This prints a score for each class. Scores are not calibrated confidence or a detector for unrelated images.

## What a training run saves

Every run gets a new output directory; existing runs cannot be overwritten. Artifacts include `best.keras`, resolved configuration and environment metadata, the exact image/split manifest, history, test classification report, confusion matrix, and per-image test probabilities. The best checkpoint is selected by validation loss; test images are evaluated after fitting.

Inputs use a shared RGB conversion, bilinear resize to 50 × 50, and float32 normalization to [0, 1]. The network retains the original 32/64/64 convolution blocks and two 128-unit dense layers with batch normalization, L2 regularization, and dropout. Augmentation operates during training only. See [baseline configuration](configs/baseline.json).

## Repository guide

- `src/coffee_classifier/`: data audit, preprocessing, model, training and prediction commands.
- `configs/`: experiment settings.
- `reports/`: data provenance and validation notes.
- `docs/`: methodology and historical dependency snapshot.
- `tests/`: checks for duplicate handling, label conflicts, corrupt images and RGB semantics.
- `training_notebook.ipynb`, `images/`: preserved historical notebook and figures.
- `data/`: existing source images and metadata.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Next experiments

Run the full cleaned baseline, review errors and near-duplicates, repeat with multiple seeds, and evaluate on genuinely new photographs. A public demo should use a validated checkpoint and state these limitations. This is an educational classifier, not a validated commercial roast measurement system.

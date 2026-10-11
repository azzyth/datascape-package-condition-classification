# DATASCAPE 2026 — Package Condition Classification

**Macro-F1 classification of e-commerce package condition (`Aman` / `Rusak` / `Lainnya`).**

> **DATASCAPE 2026** — Data Mining Competition · HIMA DS Telkom University · hosted on Kaggle
> **Public LB: 0.79539** · **OOF CV: 0.8666 ± 0.0117** (ConvNeXt-Base, 5-fold) · **rank ~42/47**

---

## The problem

Every day millions of packages move through the Indonesian e-commerce supply chain. Physical
damage costs money and erodes trust — but the QC photos coming back from couriers and hubs are
messy: heterogeneous resolutions, extreme lighting, heavy class imbalance (21 : 7 : 1), and web-scraped
noise. DATASCAPE asks for a model that classifies each package photo into **Aman** (intact),
**Rusak** (damaged), or **Lainnya** (non-standard / out-of-distribution packaging), scored by
**macro-averaged F1** so the minority classes count equally.

## What I shipped

A `ConvNeXt-Base` (ImageNet-22k → 1k pretrain, 384px) fine-tuned over 5 frozen folds, with a
deterministic near-duplicate calibration step on top. Full pipeline in
[`scripts/`](scripts/) — one audited, reproducible sequence from raw images to submission.

## The part of the story that actually mattered

The model was never the hard part. Three findings carried the whole competition:

### 1. A third of the test set was already in the train set

The data was web-scraped and full of duplicates. Perceptual-hash (pHash) grouping showed
**36.6% of test images (1,119 / 3,057) had a near-duplicate in train**, including 886
pixel-identical pairs. Worse, **122 train pairs were pixel-identical but labelled differently**
(mostly `Aman` ↔ `Rusak`).

The obvious consequence: a random train/val split would leak duplicates across the split and lie
to me. So the validation was built *first*: `StratifiedGroupKFold` grouped by pHash cluster, with
the folds frozen to a file before any model was trained. Every experiment after that was compared
on the same assignment — so a 0.001 score difference was a real difference, not noise.

### 2. The best model on my validation lost on the leaderboard

I tested every blend. `convnext_base + convnext_tiny + DINOv2-kNN` scored **0.8737 OOF** — the
highest out-of-fold. Submitted, it dropped to **0.77027**, below the single base model's
**0.79539**.

The failure analysis was the interesting part: the blend flipped 128 test labels versus the solo
model (96 net `Rusak → Aman`), concentrated on exactly the low-confidence `Aman`/`Rusak` boundary
where the labels were noisiest. On OOF those flips were net-correct; on the real test they were
net-wrong. Classic overfitting to a noisy decision boundary. The documented lesson: **the model
that wins OOF doesn't win the test set when the test set itself has label noise** — so the final
submission was the strongest *single* model, not the fanciest blend.

### 3. Grad-CAM showed the model wasn't cheating — the labels were

Adversarial validation gave AUC **0.866** for "train vs test", meaning strong distribution drift.
The biggest separators weren't content but *compression artifacts* (bits-per-pixel, file size,
resolution) — so a model could "cheat" by latching onto JPEG quality instead of package condition.
I attacked that with domain-specific augmentation (random JPEG compression, random downscale).

Then Grad-CAM confirmed the diagnosis: on correct predictions the heatmap sat on the package; on
errors it *still* sat on the package rather than being fooled by watermarks or logos. The mistakes
came from visually ambiguous or mislabelled objects — not shortcut features. When the object is
right but the label is wrong, the ceiling is the data, not the model.

### The fix that cost nothing but was worth +0.008

Because so many test images were copies of train images, a **deterministic lookup** — "if this
test image belongs to a pHash group that is 100% one label in train, use that label" — replaced the
model's guess for 958 images (58 flips). It's not inference, it's a data-structure lookup, and it
only touched groups with *unanimous* labels. Consistent, measurable, boring, effective.

## Results

| Submission | Model | OOF macro-F1 | Public LB |
|---|---|---|---|
| baseline | LogisticRegression (mean-pixel) | 0.2983 | 0.29522 |
| #2 | ResNet-18 + TTA | 0.7847 | 0.74324 |
| #3 | ConvNeXt-Tiny @512 | 0.8464 | 0.77186 |
| #4 | ConvNeXt-Tiny + near-dup | 0.8464 | 0.77968 |
| #5 | blend3 (base+tiny+DINOv2-kNN) | **0.8737** | 0.77027 |
| **#6 final** | **ConvNeXt-Base + near-dup** | 0.8666 | **0.79539** |

Per-class F1 (OOF): **Aman 0.92 · Rusak 0.76 · Lainnya 0.86**. Majority-class baseline: 0.2790.

## Why the CV → LB gap is a feature, not a bug

OOF 0.8666 vs LB 0.7954 is a ~0.07 gap. Given the label-noise finding, that gap is *expected*:
the model is partly scored against wrong labels. I chose the final model on CV (stable) rather than
chasing public LB (only 30% of test, noisier).

## Reproduce

The [`scripts/`](scripts/) pipeline is self-contained and auto-detects the dataset location
(`DATASCAPE_BASE` → local Windows path → `/kaggle/input`). The single-file entry point used inside
the Kaggle notebook is [`scripts/kaggle_all_in_one.py`](scripts/kaggle_all_in_one.py):

```bash
pip install -r requirements.txt
python scripts/kaggle_all_in_one.py --stage audit      # integrity + corrupt check
python scripts/kaggle_all_in_one.py --stage phash      # near-duplicate grouping
python scripts/kaggle_all_in_one.py --stage folds      # frozen StratifiedGroupKFold
python scripts/kaggle_all_in_one.py --stage smoke      # fast sanity pass
python scripts/kaggle_all_in_one.py --stage train      # needs GPU
python scripts/kaggle_all_in_one.py --stage ensemble   # fold-mean + near-dup calibration
python scripts/kaggle_all_in_one.py --stage submit     # build + assert submission
```

Seed `42`, folds frozen and hash-locked, every eval transform identical between val and test.

## Structure

```
datascape_notebook.ipynb      # Kaggle submission notebook (entry point)
scripts/                      # the pipeline: audit → phash → folds → train → ensemble → submit
figures/                      # EDA grids + Grad-CAM overlay
writeup.pdf                   # official competition write-up (Bahasa Indonesia)
sample_submission.csv         # expected submission format
```

## Reproducibility & AI declaration

- **External data:** none — competition data only.
- **Pretrained weights:** `convnext_base.fb_in22k_ft_in1k_384`, `convnext_tiny`, ResNet-18 (ImageNet), DINOv2 (probe only, not in final).
- **AI usage:** an LLM assistant was used for pair-programming and documentation. **No AI/VLM produced any test prediction** — every label comes from the trained CNN (5-fold mean) plus the deterministic near-duplicate lookup.

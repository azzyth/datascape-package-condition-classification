# DATASCAPE 2026 — Pipeline Scripts

Semua kode tahap disimpan sebagai `.py` self-contained supaya mudah dipindah ke
Jupyter/Kaggle. Setiap script auto-detect lokasi dataset:

1. Env var `DATASCAPE_BASE` (kalau diset)
2. Path lokal Windows (default)
3. `/kaggle/input/*/Train_Set` atau `/kaggle/input/*/*/Train_Set` (Kaggle notebook)

## Cara pakai di Kaggle
1. Upload `Train_Set/` + `Test_Set/` (+ `sample_submission.csv`) sebagai **private dataset**
   (atau attach competition data kalau tersedia).
2. Notebook baru → GPU off (stage 1–4 CPU saja).
3. Taruh file `.py` stage di `/kaggle/working` (upload via notebook editor atau git),
   lalu jalankan cell: `%run stage1_audit.py` dst. Artefak muncul di
   `<BASE>/artifacts/` — di Kaggle, set `DATASCAPE_BASE=/kaggle/working` supaya
   artefak ikut tersimpan sebagai output notebook.

## Peta script
| File | Stage | Output utama |
|---|---|---|
| `stage1_audit.py` | Load + audit integritas | `artifacts/audit_report.csv` |
| `stage1_baseline.py` | Fast baseline + submit | `artifacts/cheap_features_*.csv`, `artifacts/submission_baseline_logreg.csv` |
| `stage2_phash.py` | pHash grouping | `artifacts/phash_groups.csv`, `artifacts/image_groups.csv` (+ montase verifikasi) |
| `stage3_folds.py` | Validation strategy | `artifacts/folds.csv` (StratifiedGroupKFold by pHash group, BEKUKAN) |
| `stage5_train_kaggle.py` *(next)* | Fine-tune resnet18 384px | butuh **Kaggle Cloud GPU** (T4x2/P100), output OOF + test probs |
| `stage7_ensemble.py` | Ensemble/TTA (J4) | baca `artifacts/preds/*_oof.npz` → `ensemble_report.json` + `preds/ensemble/test_probs_blend.npz` |
| `stage9_submit.py` | Submission & assert (J4) | mode validate/build/submit; assert vs sample_submission + log `submission_log.csv` |
| `kaggle_all_in_one.py` | SEMUA (gabungan) | **untuk Kaggle GPU notebook**: 1 file = stage 1–9; `%run kaggle_all_in_one.py --stage smoke/train/...`; hash-lock otomatis vs artefak lokal |

## KONTRAK INPUT stage7 (diproduksi stage5 / J3)
`artifacts/preds/<nama>_oof.npz` dan `<nama>_test.npz`, masing-masing keys:
`ids` (str) + `probs` (float32 [n,3]). `ids` oof = urutan folds.csv, `ids` test =
urutan sample_submission.csv. Kelas Aman=0/Rusak=1/Lainnya=2.

## Konvensi
- Kelas: `Aman=0, Rusak=1, Lainnya=2` (urutan alfabetis).
- Seed 42, fold dibekukan di `artifacts/folds.csv` sejak CV pertama.
- Semua transform eval identik untuk validasi & test (test-time parity).
- GPU training HANYA di Kaggle cloud (SAC Windows memblok CUDA torch lokal).

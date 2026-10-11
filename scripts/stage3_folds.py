"""DATASCAPE 2026 — Stage 3: VALIDATION STRATEGY & OOF SETUP (GATE-1).
Role (ORCHESTRATION.md): J3 Validation & ML (DeepSeek V4 Pro) — Tahap 3.
Workflow (master_prompt_cnn_computer_vision.md): Tahap 3 + Lampiran A.5.

Deliverable (DIBEKUKAN — semua eksperimen model WAJIB pakai ini):
  artifacts/folds.csv   kolom: image_id, label, target, group, fold
  - image_id == filename (dataset tanpa id terpisah; == sample_submission.filename)
  - label    : nama kelas (Aman/Rusak/Lainnya)
  - target   : int 0/1/2
  - group    : group_id pHash (root filename; singleton = filename sendiri)
  - fold     : 0..4 dari StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

Skema CV: StratifiedGroupKFold by pHash group (near-duplicate leakage guard).
Seed 42. Baseline "titik nol" = majority macro-F1 (harus dilampaui semua model).

Portable: auto-detect path lokal (Windows) atau Kaggle (/kaggle/input).
Kaggle notebook: %run stage3_folds.py   (CPU saja; tidak butuh GPU).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

# ----------------------------- konfigurasi -----------------------------
def _find_base() -> Path:
    cands = []
    if os.environ.get("DATASCAPE_BASE"):
        cands.append(Path(os.environ["DATASCAPE_BASE"]))
    cands.append(Path(r"C:/Users/PONGO AMD/Documents/codingstuff/KAGGLE_COMPETITIONS/datascape"))
    ki = Path("/kaggle/input")
    if ki.is_dir():
        cands += [p.parent for p in list(ki.glob("*/Train_Set")) + list(ki.glob("*/*/Train_Set"))]
    for c in cands:
        if (c / "Train_Set").is_dir() and (c / "Test_Set").is_dir():
            return c
    raise RuntimeError("Dataset base tidak ditemukan. Set env DATASCAPE_BASE=<path berisi Train_Set & Test_Set>.")



def _writable_art(base):
    """artifacts/ yang bisa ditulis. /kaggle/input (Kaggle) READ-ONLY ->
    redirect ke /kaggle/working/artifacts; Colab -> /content/working/artifacts."""
    from pathlib import Path as _P
    art = base / "artifacts"
    try:
        art.mkdir(parents=True, exist_ok=True)
        probe = art / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return art
    except OSError:
        out = _P("/kaggle/working") if _P("/kaggle/working").is_dir() else _P("/content/working")
        art2 = out / "artifacts"
        art2.mkdir(parents=True, exist_ok=True)
        print(f"[path] {base} read-only -> artefak ditulis ke {art2}")
        return art2


BASE = _find_base()
ART = _writable_art(BASE)

CLASSES = ["Aman", "Rusak", "Lainnya"]          # Aman=0, Rusak=1, Lainnya=2
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
N_SPLITS = 5
SEED = 42

# ----------------------------- metrik (manual, divalidasi vs sklearn) -----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 manual: rata-rata F1 per kelas (F1 = 2PR/(P+R)), skip kelas kosong."""
    n = len(CLASSES)
    out = 0.0
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out += f1
    return out / n

# ----------------------------- main -----------------------------
def main():
    # 1) muat image_groups.csv (input dari Stage 2), ambil train saja
    ig = pd.read_csv(ART / "image_groups.csv", dtype=str)
    tr = ig[ig.split == "train"].copy().reset_index(drop=True)
    assert len(tr) == 12224, f"train rows != 12224: {len(tr)}"
    y = tr.label.map(CLS2IDX).values
    groups = tr.group_id.values
    print(f"train: {len(tr)} | kelas: {tr.label.value_counts().to_dict()}")

    # 2) statistik grup
    vc = tr.groupby("group_id").size()
    n_grp = len(vc)
    n_multi = int((vc > 1).sum())
    n_img_multi = int(vc[vc > 1].sum())
    n_mixed = int(tr.groupby("group_id")["label"].nunique().gt(1).sum())
    print(f"grup: {n_grp} total | multi-anggota {n_multi} ({n_img_multi} gambar) | campur-label {n_mixed}")

    # 3) split StratifiedGroupKFold (GATE-1)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = np.full(len(tr), -1, dtype=int)
    for f, (_, va) in enumerate(sgkf.split(tr, y, groups=groups)):
        fold_of[va] = f
    assert (fold_of >= 0).all(), "ada gambar tanpa fold"

    # 4) tulis folds.csv (DIBEKUKAN)
    out = pd.DataFrame({
        "image_id": tr.filename.values,
        "label": tr.label.values,
        "target": y,
        "group": groups,
        "fold": fold_of,
    })
    out.to_csv(ART / "folds.csv", index=False)
    print(f"\nfolds.csv ditulis: {len(out)} baris | {ART / 'folds.csv'}")

    # 5) LEAKAGE GUARD
    assert out.fold.isin(range(N_SPLITS)).all(), "fold di luar 0..4"
    n_folds_per_group = out.groupby("group")["fold"].nunique()
    bad = n_folds_per_group[n_folds_per_group > 1]
    assert len(bad) == 0, f"LEAKAGE: {len(bad)} grup pHash tercampur antar fold"
    print("leakage guard PASS: tidak ada grup pHash yang menyebrang fold")

    # 6) distribusi per fold (stratifikasi)
    print("\n=== DISTRIBUSI FOLD ===")
    for f in range(N_SPLITS):
        yf = y[fold_of == f]
        d = {CLASSES[c]: int((yf == c).sum()) for c in range(len(CLASSES))}
        print(f"  fold {f}: n={len(yf):5d} {d}")

    # 7) baseline "titik nol" — majority class, macro-F1 manual & per fold
    maj = int(np.bincount(y).argmax())
    maj_pred = np.full_like(y, maj)
    manual = macro_f1(y, maj_pred)
    sk = f1_score(y, maj_pred, average="macro")
    print(f"\n=== BASELINE (titik nol) ===")
    print(f"majority = '{CLASSES[maj]}' | macro-F1 manual={manual:.4f} | sklearn={sk:.4f} (match={abs(manual-sk)<1e-9})")
    per_fold = []
    for f in range(N_SPLITS):
        m = f1_score(y[fold_of == f], np.full((fold_of == f).sum(), maj), average="macro")
        per_fold.append(m)
    per_fold = np.array(per_fold)
    print(f"majority macro-F1 per fold: {per_fold.round(4).tolist()} | mean={per_fold.mean():.4f} ± std={per_fold.std():.4f}")

    # 8) validasi metrik manual vs sklearn pada prediksi non-trivial (dummy deterministik)
    rng = np.random.RandomState(SEED)
    dummy = rng.choice(len(CLASSES), size=len(y), p=[0.72, 0.245, 0.035])
    m1, m2 = macro_f1(y, dummy), f1_score(y, dummy, average="macro")
    assert abs(m1 - m2) < 1e-9, f"macro_f1 manual salah: {m1} vs {m2}"
    print(f"validasi metrik manual vs sklearn (prediksi acak): {m1:.4f} == {m2:.4f} OK")

    # 9) laporan
    rep = f"""# DATASCAPE 2026 — Stage 3 Report: Validation Strategy (GATE-1)

## Skema CV
- StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}) by `group_id` pHash
- Grup: {n_grp} total | multi-anggota {n_multi} ({n_img_multi} gambar) | campur-label {n_mixed}
- Campur-label dibiarkan utuh dalam satu fold (no leakage); stratifikasi pakai label mayoritas grup
- `folds.csv` DIBEKUKAN — semua eksperimen model berikutnya WAJIB memakai assignment ini

## Kolom folds.csv
`image_id` (== filename, == sample_submission.filename), `label` (Aman/Rusak/Lainnya),
`target` (0/1/2), `group` (group_id pHash), `fold` (0..4)

## Baseline titik nol
- Majority class ('{CLASSES[maj]}') macro-F1 = {manual:.4f} (manual == sklearn)
- Per fold: {per_fold.round(4).tolist()} → mean {per_fold.mean():.4f} ± {per_fold.std():.4f}
- Semua model CNN wajib jauh di atas {manual:.4f} (referensi LogReg OOF 0.2983, LB 0.29522)

## Leakage guard
- PASS: tidak ada grup pHash yang menyebrang fold
- Stratifikasi per fold seimbang (lihat distribusi di log run)
"""
    (ART / "stage3_report.md").write_text(rep, encoding="utf-8")
    print(f"\nreport tersimpan: {ART / 'stage3_report.md'}")
    print("=== STAGE 3 SELESAI — folds.csv DIBEKUKAN ===")


if __name__ == "__main__":
    main()

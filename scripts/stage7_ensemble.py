#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 7 — ENSEMBLING & TTA (J4)  —  DATASCAPE 2026
====================================================
Semua bobot dioptimasi di **OOF**, bukan di prediksi train (overfit) dan bukan
di test (tanpa label).  [master_prompt TAHAP 7 + Lampiran A.8 + GATE-3]

KONTRAK INPUT (yang HARUS diproduksi J3 — stage5_train_kaggle.py):
  Satu direktori kandidat per model:
    artifacts/preds/<nama>_oof.npz   -> keys: ids (str[n_train]), probs (float32[n_train,3])
    artifacts/preds/<nama>_test.npz  -> keys: ids (str[n_test]),  probs (float32[n_test,3])
  * ids_oof  HARUS sama persis (urutan) dengan baris artifacts/folds.csv
  * ids_test HARUS sama persis (urutan) dengan baris sample_submission.csv
  * probs = softmax 3 kelas, urutan Aman=0, Rusak=1, Lainnya=2

KONTRAK OUTPUT:
  artifacts/preds/ensemble/    -> test_probs_blend.npz (probs final untuk submission)
  artifacts/ensemble_report.json -> metrik per kandidat, bobot, keputusan SERI/tidak

ATURAN HARD:
  - Blend HANYA dari oof_probs (bukan prediksi train).
  - Gain blend < std fold -> SERI -> pilih konfigurasi paling sederhana (GATE-3).
  - Class-scaling HANYA kalau gain >= 0.02 (Tahap 7.3).
  - TTA = kandidat TERPISAH (Tahap 7.4), bukan default otomatis.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score
except ImportError:  # fallback manual macro-F1
    def f1_score(y_true, y_pred, average="macro"):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        fs = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            fs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return float(np.mean(fs)) if fs else 0.0

CLASSES = ["Aman", "Rusak", "Lainnya"]  # Aman=0, Rusak=1, Lainnya=2
N_CLASSES = len(CLASSES)
SEED = 42




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


def resolve_base():
    """Auto-detect lokasi dataset (sama seperti stage lain)."""
    env = os.environ.get("DATASCAPE_BASE")
    if env:
        return env
    # /kaggle/input READ-ONLY -> base output = /kaggle/working
    if os.path.isdir("/kaggle/input"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return os.environ.get("DATASCAPE_OUT", "/content/working")
    # fallback: direktori tempat script ini berada, naik satu level
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def load_oof_preds(base):
    """Muat semua kandidat *_oof.npz di artifacts/preds/."""
    pred_dir = os.path.join(base, "artifacts", "preds")
    cands = {}
    if not os.path.isdir(pred_dir):
        return cands
    for fn in sorted(os.listdir(pred_dir)):
        if fn.endswith("_oof.npz"):
            name = fn[: -len("_oof.npz")]
            cands[name] = os.path.join(pred_dir, fn)
    return cands


def align_to_folds(oof_path, folds):
    """Pastikan ids_oof == urutan folds.csv; return probs (n_train, 3)."""
    z = np.load(oof_path, allow_pickle=False)
    ids = z["ids"].astype(str)
    probs = z["probs"].astype(np.float32)
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"{oof_path}: probs shape {probs.shape}, harus (n,3)")
    if len(ids) != len(folds):
        raise ValueError(f"{oof_path}: {len(ids)} ids vs folds {len(folds)}")
    if not np.array_equal(ids, folds["image_id"].to_numpy().astype(str)):
        raise ValueError(f"{oof_path}: urutan ids TIDAK cocok folds.csv")
    return probs


def class_scaling(probs, y, gate=0.02):
    """Grid kasar pengali per kelas (kunci kelas 0 = 1.0) — Tahap 7.3."""
    base = macro_f1(y, probs.argmax(1))
    scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    best = {"gain": 0.0, "scale": [1.0, 1.0, 1.0]}
    for s1 in scales:
        for s2 in scales:
            w = np.array([1.0, s1, s2], dtype=np.float32)
            p = probs * w
            p /= p.sum(1, keepdims=True)
            f1 = macro_f1(y, p.argmax(1))
            if f1 - base > best["gain"]:
                best = {"gain": float(f1 - base), "scale": [1.0, s1, s2]}
    if best["gain"] < gate:
        print(f"  class-scaling TIDAK dipakai: gain {best['gain']:+.4f} < {gate}")
        return probs, [1.0, 1.0, 1.0], 0.0
    print(f"  class-scaling DIPAKAI: scale={best['scale']} gain {best['gain']:+.4f}")
    w = np.array(best["scale"], dtype=np.float32)
    p = probs * w
    p /= p.sum(1, keepdims=True)
    return p, best["scale"], best["gain"]


def optimize_blend(oof_list, y, min_gain=0.005):
    """Simplex blend di OOF (Lampiran A.8). Return weights."""
    from scipy.optimize import minimize
    base = macro_f1(y, np.mean(oof_list, axis=0).argmax(1))
    n = len(oof_list)

    def neg(logits):
        w = np.exp(logits - logits.max())
        w /= w.sum()
        return -macro_f1(y, sum(wi * o for wi, o in zip(w, oof_list)).argmax(1))

    res = minimize(neg, np.zeros(n), method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
    w = np.exp(res.x - res.x.max())
    w /= w.sum()
    gain = -res.fun - base
    return w, base, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="override DATASCAPE_BASE")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--scale-gate", type=float, default=0.02)
    args = ap.parse_args()

    base = args.base or resolve_base()
    folds_path = os.path.join(base, "artifacts", "folds.csv")
    if not os.path.isfile(folds_path):
        sys.exit(f"[GATE-1 GAGAL] folds.csv tidak ditemukan di {folds_path}")

    folds = pd.read_csv(folds_path, dtype={"image_id": str})
    if "label" not in folds.columns:
        sys.exit("folds.csv harus punya kolom 'label'")
    label2idx = {c: i for i, c in enumerate(CLASSES)}
    y = folds["label"].map(label2idx).to_numpy()
    fold_of = folds["fold"].to_numpy() if "fold" in folds.columns else np.zeros(len(folds), int)

    cands = load_oof_preds(base)
    if not cands:
        sys.exit(
            "[BLOKIR] belum ada artifacts/preds/*_oof.npz. "
            "Tunggu J3 (stage5_train_kaggle.py) selesai."
        )

    print(f"Kandidat ditemukan: {list(cands.keys())}")
    oofs, names = [], []
    for name, path in cands.items():
        try:
            oofs.append(align_to_folds(path, folds))
            names.append(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not oofs:
        sys.exit("[BLOKIR] tidak ada kandidat yang lolos alignment.")

    report = {"candidates": {}, "n_candidates": len(names)}
    for name, p in zip(names, oofs):
        f1 = macro_f1(y, p.argmax(1))
        # per-fold F1 untuk estimasi std
        fold_f1s = [macro_f1(y[fold_of == f], p[fold_of == f].argmax(1))
                    for f in np.unique(fold_of)]
        report["candidates"][name] = {
            "oof_macro_f1": round(float(f1), 5),
            "fold_f1_std": round(float(np.std(fold_f1s)), 5),
        }
        print(f"  {name}: OOF macro-F1 = {f1:.5f}  (std fold {np.std(fold_f1s):.5f})")

    # 1) Equal-weight fold-mean baseline
    base_probs = np.mean(oofs, axis=0)
    base_f1 = macro_f1(y, base_probs.argmax(1))
    base_fold_f1s = [macro_f1(y[fold_of == f], base_probs[fold_of == f].argmax(1))
                     for f in np.unique(fold_of)]
    base_std = float(np.std(base_fold_f1s))
    print(f"\nEqual-mean blend: OOF macro-F1 = {base_f1:.5f} (std fold {base_std:.5f})")

    # 2) Simplex blend — SERI rule (GATE-3)
    decision = "equal_mean"
    weights = np.ones(len(oofs)) / len(oofs)
    if len(oofs) > 1:
        w, w_base, gain = optimize_blend(oofs, y, min_gain=args.min_gain)
        thr = max(base_std, args.min_gain)
        if gain < thr:
            print(f"  blend SERI: gain {gain:+.4f} < std fold {base_std:.5f} "
                  f"-> pakai konfigurasi paling sederhana (equal-mean).")
            decision = "equal_mean"
        else:
            weights = w
            decision = "simplex"
            print(f"  blend DIPAKAI {np.round(weights, 3).tolist()}: "
                  f"{w_base:.5f} -> {w_base + gain:.5f}")
    blended = sum(wi * o for wi, o in zip(weights, oofs))

    # 3) Class scaling (gate >= scale_gate) — Tahap 7.3
    blended, scale, scale_gain = class_scaling(blended, y, gate=args.scale_gate)

    final_f1 = macro_f1(y, blended.argmax(1))
    print(f"\nFINAL OOF macro-F1 = {final_f1:.5f}")

    # Tulis test probs ensemble (rata-rata test probs tiap kandidat, bobot sama dgn blend)
    out_dir = os.path.join(base, "artifacts", "preds", "ensemble")
    os.makedirs(out_dir, exist_ok=True)
    test_probs = None
    test_ids = None
    for name in names:
        tp_path = cands[name].replace("_oof.npz", "_test.npz")
        if not os.path.isfile(tp_path):
            print(f"  [WARN] {name}: tidak ada _test.npz -> dilewati di rata-rata test")
            continue
        z = np.load(tp_path, allow_pickle=False)
        if test_ids is None:
            test_ids = z["ids"].astype(str)
            test_probs = np.zeros((len(test_ids), N_CLASSES), dtype=np.float64)
        if not np.array_equal(test_ids, z["ids"].astype(str)):
            raise ValueError(f"{tp_path}: urutan ids test tidak konsisten")
        test_probs += z["probs"].astype(np.float64)

    if test_probs is None:
        sys.exit("[BLOKIR] tidak ada _test.npz sama sekali -> tidak bisa buat submission.")

    test_probs /= len(names)
    # terapkan class-scaling yang sama ke test probs
    test_probs = test_probs * np.array(scale, dtype=np.float64)
    test_probs /= test_probs.sum(1, keepdims=True)

    np.savez_compressed(
        os.path.join(out_dir, "test_probs_blend.npz"),
        ids=test_ids, probs=test_probs.astype(np.float32),
    )

    report.update({
        "decision": decision,
        "weights": {n: round(float(w), 4) for n, w in zip(names, weights)},
        "class_scale": scale,
        "class_scale_gain": round(float(scale_gain), 4),
        "equal_mean_oof_f1": round(float(base_f1), 5),
        "equal_mean_fold_std": base_std,
        "final_oof_f1": round(float(final_f1), 5),
    })
    with open(os.path.join(base, "artifacts", "ensemble_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nArtefak ditulis: artifacts/preds/ensemble/test_probs_blend.npz")
    print(f"Report: artifacts/ensemble_report.json")


if __name__ == "__main__":
    main()

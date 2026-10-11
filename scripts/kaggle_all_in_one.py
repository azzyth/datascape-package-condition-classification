#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DATASCAPE 2026 — ALL-IN-ONE KAGGLE PIPELINE (gabungan stage1..stage9)
================================================================================
Satu file untuk notebook Kaggle GPU. Menggabungkan (dari scripts/):
  stage1_audit.py + stage1_baseline.py + stage2_phash.py + stage2b_verify.py
  + stage2c_eda.py + stage3_folds.py + stage5_train_kaggle.py + stage7_ensemble.py
  + stage9_submit.py

DATA KOMPETISI:
  - Kaggle notebook : /kaggle/input/competitions/datascape-2026-data-mining-competition/
  - Colab           : set DATASCAPE_BASE, atau taruh data di /content (auto-detect)
  - kagglehub cache : /root/.cache/kagglehub (auto-detect)

CARA PAKAI — notebook (Kaggle MAUPUN Colab):
  PILIHAN A (magic %run — argumen CLI-style):
    %run kaggle_all_in_one.py --stage smoke
    %run kaggle_all_in_one.py --stage train --name resnet18_384_kg
  PILIHAN B (paste seluruh file ini ke SATU cell — WAJIB set variabel dulu
             di cell terpisah sebelum run):
    STAGE = "smoke"                      # audit|phash|folds|smoke|train|ensemble|submit|all
    STAGE_ARGS = "--name resnet18_384_kg"   # opsional, string
  PILIHAN C (terminal): python kaggle_all_in_one.py --stage train

ARTIFACT LOCK (dibangun lokal oleh J2/J3 — WAJIB identik di Kaggle):
  artifacts/image_groups.csv  sha256(LF) = dea798c2848032b2fabfb76e0aaa539f54847019c59c85ef21d37dad57b6c04c
  artifacts/folds.csv         sha256(LF) = 9d5a602634def4f85bf06ecc21eaf98b96567805d8657c12c3a1f6b9cb78cea5
  (LF-normalized: \\r\\n -> \\n)

CARA PAKAI (cell pertama notebook Kaggle):
    !pip install -q imagehash timm
    %run kaggle_all_in_one.py --stage smoke      # SMOKE TEST dulu (wajib)
    %run kaggle_all_in_one.py --stage train     # training 5-fold penuh (T4/P100)
    %run kaggle_all_in_one.py --stage ensemble  # setelah >=1 model selesai
    %run kaggle_all_in_one.py --stage submit    # build+assert (submit via J4!)

# timm backbone (pretrained lebih kuat, IN-22k), mis:
    %run kaggle_all_in_one.py --stage train --arch timm:convnext_base.fb_in22k_ft_in1k_384 \
         --size 384 --class-weight-cap 3.0 --strong-aug --name convnext_base_384_in22k
# Kaggle GPU (T4x2/P100): convnext_base ~110M @384 -> batch 16 aman, num_workers 2-4.
# Estimasi: ~2-4 jam untuk 5-fold x 20 epoch (T4x2). P100 lebih cepat.

KELAS: Aman=0, Rusak=1, Lainnya=2 | SEED=42 | fold BEKU dari folds.csv
KONTRAK OUTPUT (dipakai J4 stage7/9):
  /kaggle/working/preds/<name>_oof.npz  (ids urutan folds.csv, probs float32[n,3])
  /kaggle/working/preds/<name>_test.npz (ids urutan sample_submission.csv)
================================================================================
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------- konstanta global -----------------------------
CLASSES = ["Aman", "Rusak", "Lainnya"]          # Aman=0, Rusak=1, Lainnya=2
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
N_SPLITS = 5
IMG_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_CAP = 768
PHASH_THRESHOLD = 6

COMP_DATA = Path("/kaggle/input/competitions/datascape-2026-data-mining-competition")
WORK = Path("/kaggle/working")

# hash lock (LF-normalized) — artefak lokal J2/J3
HASH_LOCK = {
    "image_groups.csv": "dea798c2848032b2fabfb76e0aaa539f54847019c59c85ef21d37dad57b6c04c",
    "folds.csv": "9d5a602634def4f85bf06ecc21eaf98b96567805d8657c12c3a1f6b9cb78cea5",
}

# fold BEKU (gzip+base64, urutan baris = train rows image_groups.csv) — fallback
# jika algoritma vendored pun menyimpang; dipakai oleh _folds_from_blob().
FOLDS_BLOB = (
    "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"
)



# ----------------------------- infrastruktur -----------------------------
def resolve_paths():
    """Kembalikan (BASE_INPUT, BASE_OUTPUT). Deteksi Kaggle / Colab / lokal."""
    # 1) Kaggle notebook (competition data attach)
    if COMP_DATA.is_dir():
        return COMP_DATA, WORK
    # 2) env var eksplisit (Colab: set DATASCAPE_BASE setelah upload/unzip data)
    if os.environ.get("DATASCAPE_BASE"):
        p = Path(os.environ["DATASCAPE_BASE"])
        if (p / "Train_Set").is_dir():
            return p, Path(os.environ.get("DATASCAPE_OUT", "/content/working")) if os.path.isdir("/content") else p
    # 3) Colab /content auto-detect (data diunggah/unzip ke /content)
    for c in [Path("/content"), Path("/content/data"), Path("/content/datascape")]:
        if (c / "Train_Set").is_dir() and (c / "Test_Set").is_dir():
            return c, Path("/content/working")
    # 4) kagglehub cache (kagglehub.competition_download)
    try:
        cands_kh = [Path("/root/.cache/kagglehub/competitions")]
        for ck in cands_kh:
            if ck.is_dir():
                for p in sorted(ck.glob("**/Train_Set"), recursive=True):
                    return p.parent, WORK
    except Exception:
        pass
    # 5) fallback lokal (Windows) — supaya file ini tetap bisa dites di luar Kaggle
    cands = [Path(r"C:/Users/PONGO AMD/Documents/codingstuff/KAGGLE_COMPETITIONS/datascape")]
    for c in cands:
        if (c / "Train_Set").is_dir() and (c / "Test_Set").is_dir():
            return c, c
    raise RuntimeError(
        "Data tidak ditemukan. Attach competition data (Kaggle), set env DATASCAPE_BASE "
        "(Colab: import os; os.environ['DATASCAPE_BASE']='/content/...'), atau unzip data ke /content."
    )


def sha256_lf(path) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def verify_hash_lock(art_dir: Path, name: str, fatal: bool = True) -> bool:
    p = art_dir / name
    if not p.is_file():
        if fatal:
            sys.exit(f"[HASH-LOCK GAGAL] {name} tidak ditemukan di {art_dir}")
        return False
    h = sha256_lf(p)
    ok = h == HASH_LOCK[name]
    if not ok and fatal:
        sys.exit(f"[HASH-LOCK GAGAL] {name}: sha256(LF)={h} != {HASH_LOCK[name]}")
    print(f"[hash-lock] {name}: {'MATCH' if ok else 'MISMATCH'} ({h[:16]}...)")
    return ok


def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


# ----------------------------- metrik -----------------------------
def macro_f1(y_true, y_pred) -> float:
    n = len(CLASSES)
    out = 0.0
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out += 2 * p * r / (p + r) if (p + r) else 0.0
    return out / n


# =====================================================================
# STAGE 1 — AUDIT (dari stage1_audit.py)
# =====================================================================
def check_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            color_mode, size = im2.mode, im2.size
        return {"path": path, "status": "ok", "color_mode": color_mode,
                "w": size[0], "h": size[1], "bytes": os.path.getsize(path)}
    except Exception as e:
        return {"path": path, "status": "corrupt", "color_mode": None,
                "w": None, "h": None,
                "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
                "err": str(e)[:120]}


def stage1_audit(base_in: Path, art: Path):
    from concurrent.futures import ThreadPoolExecutor
    print("=== STAGE 1: AUDIT ===")
    t0 = time.time()
    train_files, test_files = [], []
    for cls in CLASSES:
        d = base_in / "Train_Set" / cls
        for f in sorted(os.listdir(d)):
            train_files.append((f, cls, str(d / f)))
    for f in sorted(os.listdir(base_in / "Test_Set")):
        test_files.append((f, str(base_in / "Test_Set" / f)))

    with ThreadPoolExecutor(max_workers=8) as ex:
        tr = list(ex.map(check_image, [x[2] for x in train_files]))
        te = list(ex.map(check_image, [x[1] for x in test_files]))
    tr_map, te_map = {r["path"]: r for r in tr}, {r["path"]: r for r in te}

    rows = []
    for fname, cls, path in train_files:
        r = tr_map[path]
        rows.append({"split": "train", "filename": fname, "label": cls,
                     "status": r["status"], "color_mode": r["color_mode"],
                     "w": r["w"], "h": r["h"], "bytes": r["bytes"], "err": r.get("err", "")})
    for fname, path in test_files:
        r = te_map[path]
        rows.append({"split": "test", "filename": fname, "label": "",
                     "status": r["status"], "color_mode": r["color_mode"],
                     "w": r["w"], "h": r["h"], "bytes": r["bytes"], "err": r.get("err", "")})
    df = pd.DataFrame(rows)
    df.to_csv(art / "audit_report.csv", index=False)
    n_corrupt = int((df.status == "corrupt").sum())
    print(f"imgs: train={len(train_files)} test={len(test_files)} | corrupt={n_corrupt} | {time.time()-t0:.1f}s")
    assert n_corrupt == 0, "Ada gambar corrupt — investigasi!"
    print("STAGE 1 AUDIT OK")


# =====================================================================
# STAGE 2 — PHASH GROUPING (dari stage2_phash.py, tanpa montage)
# =====================================================================
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def stage2_phash(base_in: Path, art: Path):
    print("=== STAGE 2: PHASH GROUPING ===")
    t0 = time.time()
    try:
        import imagehash
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=True)
        import imagehash
    from concurrent.futures import ThreadPoolExecutor

    rows = []
    for cls in CLASSES:
        d = base_in / "Train_Set" / cls
        for f in sorted(os.listdir(d)):
            rows.append({"split": "train", "filename": f, "label": cls, "path": str(d / f)})
    for f in sorted(os.listdir(base_in / "Test_Set")):
        rows.append({"split": "test", "filename": f, "label": "", "path": str(base_in / "Test_Set" / f)})
    df = pd.DataFrame(rows)

    def compute_phash(path):
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=8)
            return str(h), int.from_bytes(np.packbits(h.hash.astype(bool)).tobytes(), "big")
        except Exception:
            return None, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(compute_phash, df["path"]))
    df["phash_hex"] = [r[0] for r in res]
    df["hash_u64"] = [r[1] for r in res]
    n_fail = int(df.hash_u64.isna().sum())
    assert n_fail == 0, f"{n_fail} gambar gagal pHash"
    print(f"pHash ok: {len(df)} imgs [{time.time()-t0:.1f}s]")

    h = df.hash_u64.values.astype(np.uint64)
    n = len(h)
    pairs = []
    block = 256
    for s in range(0, n, block):
        blk = h[s:s + block]
        xor = blk[:, None] ^ h[None, :]
        bts = xor.view(np.uint8).reshape(len(blk), n, 8)
        dist = _POP[bts].sum(axis=2, dtype=np.uint8)
        ii, jj = np.nonzero(dist <= PHASH_THRESHOLD)
        for a, c in zip(ii.tolist(), jj.tolist()):
            g = s + a
            if g < c:
                pairs.append((g, c, int(dist[a, c])))

    # DSU union-find (root kecil menang — deterministik)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, d in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj
    root_idx = [find(i) for i in range(n)]
    root2name = {i: df.filename.iloc[i] for i in sorted(set(root_idx))}
    df["group_id"] = [root2name[r] for r in root_idx]

    df[["filename", "split", "label", "phash_hex", "group_id"]].to_csv(art / "phash_full.csv", index=False)
    df[["filename", "split", "label", "group_id"]].to_csv(art / "image_groups.csv", index=False)

    vc = df.groupby("group_id").size()
    print(f"grup: {len(vc)} total | multi: {int((vc > 1).sum())} | "
          f"lintas-split: {int(df[df.split=='test'].group_id.isin(df[df.split=='train'].group_id).sum())} test imgs")
    print(f"STAGE 2 PHASH OK [{time.time()-t0:.1f}s] — verifikasi hash-lock image_groups.csv...")
    ok = verify_hash_lock(art, "image_groups.csv", fatal=True)
    return ok


# =====================================================================
# STAGE 3 — FOLDS (GATE-1) — SGKF vendored (sklearn 1.9.0) + blob BEKU
# =====================================================================
def _sgkf_fold_of(y, groups, n_splits=N_SPLITS, shuffle=True, random_state=SEED):
    """StratifiedGroupKFold VENDORED — algoritma scikit-learn 1.9.0 (Apache-2.0).

    Kenapa vendored: perilaku SGKF berubah antar versi sklearn, sehingga folds
    yang dibangun di Kaggle (sklearn versi lain) tidak identik dengan lokal.
    Implementasi murni numpy ini terverifikasi IDENTIK dengan sklearn 1.9.0
    pada data ini (12.224 baris) — bebas versi sklearn/numpy papan mana pun.
    """
    from collections import defaultdict
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
    n_classes = len(y_cnt)
    _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
    y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
    for class_idx, group_idx in zip(y_inv, groups_inv):
        y_counts_per_group[group_idx, class_idx] += 1
    y_counts_per_fold = np.zeros((n_splits, n_classes))
    groups_per_fold = defaultdict(set)
    if shuffle:
        perm = np.arange(len(groups_cnt))
        rng.shuffle(perm)
        y_counts_per_group = y_counts_per_group[perm]
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(perm.size)
        groups_inv = inv_perm[groups_inv]
    sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind="stable")
    for group_idx in sorted_groups_idx:
        group_y_counts = y_counts_per_group[group_idx]
        best_fold, min_eval, min_samples_in_fold = None, np.inf, np.inf
        for i in range(n_splits):
            y_counts_per_fold[i] += group_y_counts
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            if fold_eval < min_eval or (np.isclose(fold_eval, min_eval)
                                        and samples_in_fold < min_samples_in_fold):
                min_eval, min_samples_in_fold, best_fold = fold_eval, samples_in_fold, i
        y_counts_per_fold[best_fold] += group_y_counts
        groups_per_fold[best_fold].add(group_idx)
    fold_of = np.full(len(y), -1, dtype=int)
    for f in range(n_splits):
        fold_of[np.isin(groups_inv, list(groups_per_fold[f]))] = f
    assert (fold_of >= 0).all()
    return fold_of


def _folds_from_blob():
    """Dekode fold BEKU (gzip+base64) — urutan baris = train rows image_groups.csv."""
    txt = gzip.decompress(base64.b64decode(FOLDS_BLOB)).decode("utf-8")
    return np.array([int(v) for v in txt.strip().splitlines()], dtype=int)


def _write_folds(art: Path, tr: pd.DataFrame, y, fold_of) -> bool:
    out = pd.DataFrame({
        "image_id": tr.filename.values,
        "label": tr.label.values,
        "target": y,
        "group": tr.group_id.values,
        "fold": fold_of,
    })
    out.to_csv(art / "folds.csv", index=False)
    nfp = out.groupby("group")["fold"].nunique()
    assert (nfp <= 1).all(), "LEAKAGE: grup lintas fold"
    ok = sha256_lf(art / "folds.csv") == HASH_LOCK["folds.csv"]
    h = sha256_lf(art / "folds.csv")
    print(f"[hash-lock] folds.csv: {'MATCH' if ok else 'MISMATCH'} ({h[:16]}...)")
    return ok


def stage3_folds(art: Path):
    print("=== STAGE 3: FOLDS (GATE-1) ===")
    ig = pd.read_csv(art / "image_groups.csv", dtype=str)
    tr = ig[ig.split == "train"].copy().reset_index(drop=True)
    assert len(tr) == 12224, f"train rows != 12224: {len(tr)}"
    y = tr.label.map(CLS2IDX).values

    print("[folds] rebuild via SGKF vendored (algoritma sklearn 1.9.0)...")
    fold_of = _sgkf_fold_of(y, tr.group_id.values)
    print(f"  distribusi fold: {np.bincount(fold_of).tolist()} | leakage guard PASS")
    if _write_folds(art, tr, y, fold_of):
        print("STAGE 3 FOLDS OK — folds BEKU & IDENTIK dengan lokal")
        return
    print("[folds] vendored mismatch → fallback blob BEKU tertanam...")
    fold_of = _folds_from_blob()
    assert len(fold_of) == len(tr), f"blob {len(fold_of)} != train {len(tr)}"
    if _write_folds(art, tr, y, fold_of):
        print("STAGE 3 FOLDS OK (via blob) — folds BEKU & IDENTIK dengan lokal")
        return
    sys.exit("[BLOKIR] folds.csv tidak bisa direproduksi identik — STOP, jangan training.")


def ensure_folds(base_in: Path, art: Path) -> bool:
    """Gerbang training: folds.csv harus ada & hash MATCH; kalau tidak → rebuild."""
    if (art / "folds.csv").is_file() and sha256_lf(art / "folds.csv") == HASH_LOCK["folds.csv"]:
        print("[gate] folds.csv valid (hash MATCH)")
        return True
    print("[gate] folds.csv hilang/stale/hash-mismatch → rebuild deterministik...")
    if not (art / "image_groups.csv").is_file():
        stage2_phash(base_in, art)  # hash-lock image_groups fatal di dalamnya
    stage3_folds(art)
    return sha256_lf(art / "folds.csv") == HASH_LOCK["folds.csv"]


# =====================================================================
# STAGE 5 — TRAINING (dari stage5_train_kaggle.py, verbatim logic)
# =====================================================================
# ----------------------------- transform & dataset (MODULE-LEVEL) -----------------------------
# WAJIB di level modul (bukan nested di stage5_train): DataLoader num_workers>0 di Windows
# pakai spawn, yang re-import __main__ dan pickle dataset/transform. Kelas lokal tidak bisa
# di-pickle (PicklingError) — lihat log stage5_local_full.log. Mirror stage5_train_kaggle.py.
def pad_to_square(img: Image.Image, cap: int = PAD_CAP) -> Image.Image:
    """PAD ke persegi (hitam), sisi terpanjang di-cap ke `cap`. Preserve aspect (D6)."""
    img = img.convert("RGB")
    w, h = img.size
    s = cap / max(w, h) if max(w, h) > cap else 1.0
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class PadToSquare:
    """Wrapper callable picklable (wajib utk DataLoader workers di Windows/spawn)."""
    def __init__(self, cap: int = PAD_CAP):
        self.cap = cap

    def __call__(self, img: Image.Image) -> Image.Image:
        return pad_to_square(img, self.cap)


class RandomJPEGCompression:
    """Serang artefak kompresi (D8): re-encode JPEG dengan quality acak."""
    def __init__(self, p: float = 0.5, quality=(30, 95)):
        self.p, self.quality = p, quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            q = random.randint(*self.quality)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomDownscale:
    """Serang drift resolusi (D8): turunkan skala lalu biarkan resize naik kembali."""
    def __init__(self, p: float = 0.5, scale=(0.5, 0.9)):
        self.p, self.scale = p, scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            w, h = img.size
            s = random.uniform(*self.scale)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return img


class ImageDataset:
    """Map-style dataset (plain class OK utk DataLoader; tidak butuh Dataset base agar
    bebas import torch di level modul)."""
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return img
        return img, self.labels[i]


def build_model(arch="resnet18", num_classes=3, pretrained=True, dropout=0.3):
    """Backbone + head (A.2). MODULE-LEVEL: dipakai stage5_train DAN stage5_tta.

    Dua sumber:
    - torchvision: resnet18/34/50 | efficientnet_b0/b2 | convnext_tiny/small | densenet121
      (dipanggil via getattr(models, arch)(weights="DEFAULT"))
    - timm (prefix "timm:"): backbone pretrained lebih kuat, mis.
      "timm:convnext_base.fb_in22k_ft_in1k_384" (IN-22k -> IN-1k, 384px). timm.create_model
      dengan num_classes=num_classes membangun head sendiri; kita GANTI dengan head kita.
    """
    import torch.nn as nn
    if arch.startswith("timm:"):
        import timm
        name = arch[len("timm:"):]
        m = timm.create_model(name, pretrained=pretrained, num_classes=0)  # backbone saja
        feat = m.num_features
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))
        return nn.Sequential(m, head), feat
    from torchvision import models
    m = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    if arch.startswith("resnet"):
        feat = m.fc.in_features; m.fc = nn.Identity()
    elif arch.startswith("efficientnet"):
        feat = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif arch.startswith("convnext"):
        feat = m.classifier[2].in_features; m.classifier[2] = nn.Identity()
    else:  # densenet
        feat = m.classifier.in_features; m.classifier = nn.Identity()
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))
    return nn.Sequential(m, head), feat


def stage5_train(base_in: Path, art: Path, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms

    print("=== STAGE 5: TRAINING ===")
    t0 = time.time()
    set_seed(SEED)

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    use_amp = device.type == "cuda"
    print(f"[train] device={device} | amp={use_amp} | torch={torch.__version__}")
    if device.type != "cuda":
        if not args.smoke:
            sys.exit("[BLOKIR] --stage train di CPU terdeteksi — aktifkan GPU (Settings → Accelerator → "
                     "GPU T4 x2 / P100) lalu RESTART notebook, atau pakai --stage smoke.")
        print("[train] WARNING: CPU — hanya untuk smoke!")

    folds = pd.read_csv(art / "folds.csv", dtype={"image_id": str})
    if args.cache:
        cache = Path(args.cache)
        assert (cache / "Train_Set").is_dir(), f"cache tidak valid: {cache}"
        folds["path"] = [str(cache / "Train_Set" / lbl / (Path(fn).stem + ".jpg"))
                         for lbl, fn in zip(folds["label"], folds["image_id"])]
        print(f"[train] CACHE mode: {cache} (pad sudah diterapkan; PadToSquare jadi no-op)")
    else:
        folds["path"] = [str(base_in / "Train_Set" / lbl / fn)
                         for lbl, fn in zip(folds["label"], folds["image_id"])]
    y_all = folds["target"].to_numpy(dtype=np.int64)

    ss = pd.read_csv(base_in / "sample_submission.csv")
    if args.cache:
        test_paths = [str(Path(args.cache) / "Test_Set" / (Path(fn).stem + ".jpg"))
                      for fn in ss["filename"]]
    else:
        test_paths = [str(base_in / "Test_Set" / fn) for fn in ss["filename"]]
    test_ids = ss["filename"].to_numpy(dtype=str)

    if args.smoke:
        test_paths = test_paths[:args.smoke_n]
        test_ids = test_ids[:args.smoke_n]
        sub = folds.groupby("label", group_keys=False).apply(
            lambda d: d.sample(n=min(len(d), max(3, args.smoke_n // 3)), random_state=SEED)
        ).reset_index(drop=True)
        idx = np.arange(len(sub))
        tr_idx, va_idx = [], []
        for c in range(len(CLASSES)):
            ci = idx[sub["target"].to_numpy() == c]
            np.random.seed(SEED)
            np.random.shuffle(ci)
            nv = max(1, int(round(0.2 * len(ci))))
            va_idx += list(ci[:nv])
            tr_idx += list(ci[nv:])
        split_list = [(np.array(tr_idx), np.array(va_idx))]
        folds = sub.reset_index(drop=True)
        y_all = sub["target"].to_numpy(dtype=np.int64)
        args.name = f"{args.name}_smoke" if not args.name.endswith("_smoke") else args.name
        print(f"[smoke] subset n={len(sub)} (train {len(tr_idx)} / val {len(va_idx)})")
    else:
        split_list = []
        for f in range(args.folds):
            va = np.where(folds["fold"].to_numpy() == f)[0]
            tr = np.where(folds["fold"].to_numpy() != f)[0]
            split_list.append((tr, va))

    counts = np.bincount(y_all, minlength=3)
    weights = len(y_all) / (3.0 * np.maximum(counts, 1))
    if args.class_weight_cap and args.class_weight_cap > 0:
        weights = np.minimum(weights, args.class_weight_cap)
        print(f"[train] class-weight cap={args.class_weight_cap} diterapkan")
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[train] class counts={counts.tolist()} | weights={weights.round(4).tolist()}")

    _sz = args.size
    if args.strong_aug:
        # drift-robust: serang kompresi (JPEG q20-90), resolusi (downscale 0.4-0.9),
        # dan pencahayaan lebih keras (temuan adversarial validation AUC 0.866)
        train_tf = transforms.Compose([
            PadToSquare(PAD_CAP),
            RandomDownscale(p=0.6, scale=(0.4, 0.9)),
            transforms.RandomResizedCrop(size=_sz, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            RandomJPEGCompression(p=0.6, quality=(20, 90)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value=0.0),
        ])
    else:
        train_tf = transforms.Compose([
            PadToSquare(PAD_CAP),
            RandomDownscale(p=0.5, scale=(0.5, 0.9)),
            transforms.RandomResizedCrop(size=_sz, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            RandomJPEGCompression(p=0.5, quality=(30, 95)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.0),
        ])
    eval_tf = transforms.Compose([
        PadToSquare(PAD_CAP),
        transforms.Resize((_sz, _sz)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def train_one_epoch(model, loader, optimizer, criterion, scaler):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def predict(model, loader):
        model.eval()
        outs = []
        for x, _ in loader:
            x = x.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                p = torch.softmax(model(x).float(), dim=1)
            outs.append(p.cpu())
        return torch.cat(outs).numpy()

    def collapse_report(probs, y_true):
        pred = probs.argmax(1)
        dist = {CLASSES[c]: int((pred == c).sum()) for c in range(len(CLASSES))}
        f1 = macro_f1(y_true, pred)
        collapsed = len([v for v in dist.values() if v > 0]) == 1
        return dist, f1, collapsed

    n_workers = args.num_workers
    test_ds = ImageDataset(test_paths, np.zeros(len(test_paths), dtype=np.int64), eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=n_workers, pin_memory=use_amp,
                             persistent_workers=(n_workers > 0))

    (art / "ckpt").mkdir(parents=True, exist_ok=True)  # torch.save butuh parent ada (bug Kaggle: ckpt/ belum dibuat)
    n = len(folds)
    oof_probs = np.zeros((n, 3), dtype=np.float32)
    oof_ids = folds["image_id"].to_numpy(dtype=str)
    fold_f1s = []
    test_sum = None

    for fi, (tr_idx, va_idx) in enumerate(split_list):
        print(f"\n===== FOLD {fi} =====")
        tr_ds = ImageDataset(folds["path"].iloc[tr_idx].tolist(), y_all[tr_idx], train_tf)
        va_ds = ImageDataset(folds["path"].iloc[va_idx].tolist(), y_all[va_idx], eval_tf)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=n_workers, pin_memory=use_amp,
                               persistent_workers=(n_workers > 0))
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=n_workers, pin_memory=use_amp,
                               persistent_workers=(n_workers > 0))

        model, _ = build_model(arch=args.arch, pretrained=True, dropout=args.dropout)
        model.to(device)
        backbone, head = model[0], model[1]
        optimizer = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": args.lr_backbone},
            {"params": head.parameters(), "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                 total_iters=args.warmup)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cos], milestones=[args.warmup])
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_f1 = -1.0
        best_path = art / "ckpt" / f"{args.name}_fold{fi}_best.pt"
        for ep in range(args.epochs):
            loss = train_one_epoch(model, tr_loader, optimizer, criterion, scaler)
            scheduler.step()
            vp = predict(model, va_loader)
            dist, f1, collapsed = collapse_report(vp, y_all[va_idx])
            flag = "  <-- COLLAPSE!" if collapsed else ""
            print(f"  ep {ep}: loss={loss:.4f} val_macroF1={f1:.4f} dist={dist}{flag}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        vp = predict(model, va_loader)
        _, best_f1, _ = collapse_report(vp, y_all[va_idx])
        oof_probs[va_idx] = vp
        fold_f1s.append(best_f1)
        print(f"  fold {fi} BEST val macro-F1 = {best_f1:.4f}")
        tp = predict(model, test_loader)
        test_sum = tp if test_sum is None else test_sum + tp
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    test_probs = test_sum / len(split_list)
    fold_f1s = np.array(fold_f1s)
    cv_mean, cv_std = float(fold_f1s.mean()), float(fold_f1s.std())
    oof_best = macro_f1(y_all, oof_probs.argmax(1))
    print(f"\n===== HASIL =====\n  fold F1: {fold_f1s.round(4).tolist()}\n"
          f"  CV macro-F1 = {cv_mean:.4f} ± {cv_std:.4f} | OOF = {oof_best:.4f}")

    # KONTRAK J4 (di /kaggle/working/preds)
    preds_dir = WORK / "preds" if WORK.is_dir() and str(art) == str(COMP_DATA) else art / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preds_dir / f"{args.name}_oof.npz", ids=oof_ids, probs=oof_probs.astype(np.float32))
    np.savez_compressed(preds_dir / f"{args.name}_test.npz", ids=test_ids, probs=test_probs.astype(np.float32))
    print(f"[save] {preds_dir / (args.name + '_oof.npz')}")
    print(f"[save] {preds_dir / (args.name + '_test.npz')}")
    od = {CLASSES[c]: int((oof_probs.argmax(1) == c).sum()) for c in range(len(CLASSES))}
    print(f"[sanity] OOF dist: {od} (train ~ 72/24.6/3.5)")

    if not args.smoke:
        # metrics_summary di working (append, J1 miliknya tapi stage5 boleh append per kontrak)
        mrow = pd.DataFrame([{
            "experiment": args.name, "stage": 5,
            "cv_mean": round(cv_mean, 4), "cv_std": round(cv_std, 4),
            "oof_best": round(oof_best, 4), "lb_public": "", "lb_private": "",
            "note": f"{device.type}; {args.arch}@{args.size}; "
                    f"aug={'strong' if args.strong_aug else 'std'}; "
                    f"cw_cap={args.class_weight_cap}; classweight+LS0.1; fold-mean",
            "date": datetime.now().strftime("%Y-%m-%d")}])
        mp = preds_dir.parent / "metrics_summary.csv"
        if mp.exists():
            pd.concat([pd.read_csv(mp), mrow], ignore_index=True).to_csv(mp, index=False)
        else:
            mrow.to_csv(mp, index=False)
    print(f"TOTAL RUNTIME {time.time()-t0:.0f}s")


# =====================================================================
# STAGE 5b — TTA (hflip) INFERENCE (J4, Tahap 7.4) — kandidat TERPISAH
# =====================================================================
def stage5_tta(base_in: Path, art: Path, args):
    """Muat ckpt per fold (--name), infer ulang OOF + test dengan hflip TTA.
    Output: <name>_oof_hflip.npz + <name>_test_hflip.npz (kandidat TTA terpisah).
    Print gain OOF(hflip) vs OOF(tanpa TTA) supaya J4 bisa putuskan dipakai/tidak."""
    print("=== STAGE 5b: TTA (hflip) INFERENCE (Tahap 7.4) ===")
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[tta] device={device} | torch={torch.__version__}")

    folds = pd.read_csv(art / "folds.csv", dtype={"image_id": str})
    if args.cache:
        cache = Path(args.cache)
        folds["path"] = [str(cache / "Train_Set" / lbl / (Path(fn).stem + ".jpg"))
                         for lbl, fn in zip(folds["label"], folds["image_id"])]
    else:
        folds["path"] = [str(base_in / "Train_Set" / lbl / fn)
                         for lbl, fn in zip(folds["label"], folds["image_id"])]
    ss = pd.read_csv(base_in / "sample_submission.csv")
    if args.cache:
        test_paths = [str(Path(args.cache) / "Test_Set" / (Path(fn).stem + ".jpg"))
                      for fn in ss["filename"]]
    else:
        test_paths = [str(base_in / "Test_Set" / fn) for fn in ss["filename"]]
    test_ids = ss["filename"].to_numpy(dtype=str)

    eval_tf = transforms.Compose([
        PadToSquare(PAD_CAP),
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    nw = args.num_workers
    test_ds = ImageDataset(test_paths, None, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=use_amp,
                             persistent_workers=(nw > 0))

    @torch.no_grad()
    def predict_tta(model, loader):
        model.eval()
        outs = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            views = [x, torch.flip(x, dims=[3])]  # hflip (A.6)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                p = sum(torch.softmax(model(v).float(), dim=1) for v in views) / len(views)
            outs.append(p.cpu())
        return torch.cat(outs).numpy()

    y_all = folds["label"].map(CLS2IDX).to_numpy()
    oof_probs = np.zeros((len(folds), 3), dtype=np.float32)
    oof_ids = folds["image_id"].to_numpy(dtype=str)
    test_sum = np.zeros((len(test_ids), 3), dtype=np.float64)

    for fi in range(args.folds):
        ckpt = art / "ckpt" / f"{args.name}_fold{fi}_best.pt"
        if not ckpt.is_file():
            sys.exit(f"[BLOKIR] {ckpt} tidak ada — training {args.name} belum selesai.")
        model, _ = build_model(arch=args.arch, pretrained=True, dropout=args.dropout)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        va_idx = np.where(folds["fold"].to_numpy() == fi)[0]
        va_ds = ImageDataset(folds["path"].iloc[va_idx].tolist(), y_all[va_idx], eval_tf)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=nw, pin_memory=use_amp,
                               persistent_workers=(nw > 0))
        oof_probs[va_idx] = predict_tta(model, va_loader)
        test_sum += predict_tta(model, test_loader)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_probs = (test_sum / args.folds).astype(np.float32)
    oof_tta = macro_f1(y_all, oof_probs.argmax(1))
    print(f"[tta] OOF macro-F1 (hflip) = {oof_tta:.5f}")

    # bandingkan vs OOF non-TTA (dari training) — gain = dasar keputusan J4
    base_path = art / "preds" / f"{args.name}_oof.npz"
    if base_path.is_file():
        z = np.load(base_path, allow_pickle=False)
        base_f1 = macro_f1(y_all, z["probs"].argmax(1))
        print(f"[tta] OOF tanpa-TTA = {base_f1:.5f} | gain TTA = {oof_tta - base_f1:+.5f}")
    else:
        print("[tta] (tidak ada *_oof.npz non-TTA utk dibandingkan)")

    preds_dir = art / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preds_dir / f"{args.name}_oof_hflip.npz",
                        ids=oof_ids, probs=oof_probs.astype(np.float32))
    np.savez_compressed(preds_dir / f"{args.name}_test_hflip.npz",
                        ids=test_ids, probs=test_probs)
    print(f"[save] {preds_dir / (args.name + '_oof_hflip.npz')}")
    print(f"[save] {preds_dir / (args.name + '_test_hflip.npz')}")


# =====================================================================
# STAGE 7 — ENSEMBLE (dari stage7_ensemble.py)
# =====================================================================
def stage7_ensemble(base_in: Path, art: Path, args):
    print("=== STAGE 7: ENSEMBLE (GATE-3) ===")
    from scipy.optimize import minimize

    # cari preds di art/preds atau WORK/preds
    pred_dirs = [art / "preds", WORK / "preds"]
    cands = {}
    for pd_ in pred_dirs:
        if pd_.is_dir():
            for fn in sorted(os.listdir(pd_)):
                if fn.endswith("_oof.npz"):
                    cands[fn[:-len("_oof.npz")]] = pd_ / fn
    if not cands:
        sys.exit("[BLOKIR] belum ada *_oof.npz — jalankan --stage train dulu.")

    folds = pd.read_csv(art / "folds.csv", dtype={"image_id": str})
    y = folds["label"].map(CLS2IDX).to_numpy()
    fold_of = folds["fold"].to_numpy()

    def align(oof_path):
        z = np.load(oof_path, allow_pickle=False)
        ids, probs = z["ids"].astype(str), z["probs"].astype(np.float32)
        assert probs.shape == (len(folds), 3), f"{oof_path}: shape {probs.shape}"
        assert np.array_equal(ids, folds["image_id"].to_numpy().astype(str)), \
            f"{oof_path}: urutan ids != folds.csv"
        return probs

    oofs, names = [], []
    for name, path in cands.items():
        try:
            oofs.append(align(path)); names.append(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    if not oofs:
        sys.exit("[BLOKIR] tidak ada kandidat yang lolos alignment "
                 "(npz smoke tidak lolos — jalankan --stage train penuh dulu).")
    print(f"kandidat: {names}")

    report = {"candidates": {}}
    for name, p in zip(names, oofs):
        f1 = macro_f1(y, p.argmax(1))
        ff = [macro_f1(y[fold_of == f], p[fold_of == f].argmax(1)) for f in np.unique(fold_of)]
        report["candidates"][name] = {"oof_macro_f1": round(float(f1), 5),
                                      "fold_f1_std": round(float(np.std(ff)), 5)}
        print(f"  {name}: OOF={f1:.5f} (std fold {np.std(ff):.5f})")

    base_probs = np.mean(oofs, axis=0)
    base_f1 = macro_f1(y, base_probs.argmax(1))
    ff = [macro_f1(y[fold_of == f], base_probs[fold_of == f].argmax(1)) for f in np.unique(fold_of)]
    base_std = float(np.std(ff))
    print(f"equal-mean blend: OOF={base_f1:.5f} (std {base_std:.5f})")

    decision, weights = "equal_mean", np.ones(len(oofs)) / len(oofs)
    if len(oofs) > 1:
        def neg(logits):
            w = np.exp(logits - logits.max()); w /= w.sum()
            return -macro_f1(y, sum(wi * o for wi, o in zip(w, oofs)).argmax(1))
        res = minimize(neg, np.zeros(len(oofs)), method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
        w = np.exp(res.x - res.x.max()); w /= w.sum()
        gain = -res.fun - base_f1
        if gain < max(base_std, args.min_gain):
            print(f"  blend SERI (gain {gain:+.4f} < std {base_std:.5f}) -> equal-mean")
        else:
            weights, decision = w, "simplex"
            print(f"  blend DIPAKAI {np.round(w,3).tolist()}: {base_f1:.5f} -> {base_f1+gain:.5f}")
    blended = sum(wi * o for wi, o in zip(weights, oofs))

    # class scaling (gate >= 0.02)
    scale = [1.0, 1.0, 1.0]
    base2 = macro_f1(y, blended.argmax(1))
    best = {"gain": 0.0, "scale": [1.0, 1.0, 1.0]}
    for s1 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]:
        for s2 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]:
            wv = np.array([1.0, s1, s2], dtype=np.float32)
            p = blended * wv; p /= p.sum(1, keepdims=True)
            f1 = macro_f1(y, p.argmax(1))
            if f1 - base2 > best["gain"]:
                best = {"gain": float(f1 - base2), "scale": [1.0, s1, s2]}
    if best["gain"] >= args.scale_gate:
        scale = best["scale"]
        print(f"  class-scaling DIPAKAI {scale} gain {best['gain']:+.4f}")
    else:
        print(f"  class-scaling skip (gain {best['gain']:+.4f} < {args.scale_gate})")
    wv = np.array(scale, dtype=np.float32)
    blended = blended * wv; blended /= blended.sum(1, keepdims=True)
    final_f1 = macro_f1(y, blended.argmax(1))
    print(f"FINAL OOF = {final_f1:.5f}")

    # test probs blend
    test_ids, test_sum = None, None
    for name in names:
        tp = cands[name].with_name(cands[name].name.replace("_oof.npz", "_test.npz"))
        if not tp.is_file():
            print(f"  [WARN] {name}: _test.npz hilang"); continue
        z = np.load(tp, allow_pickle=False)
        if test_ids is None:
            test_ids = z["ids"].astype(str)
            test_sum = np.zeros((len(test_ids), 3), dtype=np.float64)
        test_sum += z["probs"].astype(np.float64)
    if test_sum is None:
        sys.exit("[BLOKIR] tidak ada _test.npz")
    test_sum /= len(names)
    test_sum = test_sum * wv
    test_sum /= test_sum.sum(1, keepdims=True)

    out_dir = (art / "preds" / "ensemble") if (art / "preds").is_dir() else WORK / "preds" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "test_probs_blend.npz",
                        ids=test_ids, probs=test_sum.astype(np.float32))
    report.update({"decision": decision,
                   "weights": {n: round(float(w), 4) for n, w in zip(names, weights)},
                   "class_scale": scale, "final_oof_f1": round(float(final_f1), 5)})
    with open(art / "ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[save] {out_dir / 'test_probs_blend.npz'} | report: ensemble_report.json")


# =====================================================================
# STAGE 9 — SUBMISSION BUILD/ASSERT/SUBMIT (dari stage9_submit.py)
# =====================================================================
SUBMISSION_LOG_COLS = ["submission_id", "filename", "date", "stage", "model",
                       "description", "cv_score", "oof_score", "lb_public",
                       "lb_private", "note"]


def _log_submission(art: Path, candidate: str, cv: str = "", oof: str = "",
                    filename: str = "", note: str = "", status: str = ""):
    """Append baris ke artifacts/submission_log.csv (skema milik J1; J4 append-only)."""
    import csv
    log_path = art / "submission_log.csv"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {"submission_id": "", "filename": filename, "date": now, "stage": 9,
           "model": candidate, "description": note, "cv_score": cv,
           "oof_score": oof, "lb_public": "", "lb_private": "", "note": status}
    exists = log_path.is_file()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_LOG_COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[log] {log_path}")


def _kaggle_submit(sub_path: Path, message: str):
    """Upload via `python -m kaggle competitions submit` (butuh OAuth credentials.json).
    `kaggle` CLI tidak ada di PATH — pakai sys.executable -m kaggle (venv)."""
    cmd = [sys.executable, "-m", "kaggle", "competitions", "submit", "-c",
           "datascape-2026-data-mining-competition", "-f", str(sub_path), "-m", message]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def stage9_submit(base_in: Path, art: Path, args):
    print("=== STAGE 9: SUBMISSION BUILD + ASSERT (GATE-4) ===")
    TRAIN_PROP = {"Aman": 0.720, "Rusak": 0.245, "Lainnya": 0.035}
    ss = pd.read_csv(base_in / "sample_submission.csv", keep_default_na=False)

    probs_path = Path(args.probs) if args.probs else art / "preds" / "ensemble" / "test_probs_blend.npz"
    if not probs_path.is_file():
        alt = WORK / "preds" / "ensemble" / "test_probs_blend.npz"
        if alt.is_file():
            probs_path = alt
        else:
            sys.exit(f"[BLOKIR] {probs_path} tidak ada — jalankan --stage ensemble dulu.")
    z = np.load(probs_path, allow_pickle=False)
    ids, probs = z["ids"].astype(str), z["probs"]
    pred = probs.argmax(1)
    sub = pd.DataFrame({"filename": ids, "label": [CLASSES[i] for i in pred]})

    errs = []
    if list(sub.columns) != ["filename", "label"]:
        errs.append("kolom salah")
    if len(sub) != len(ss):
        errs.append(f"baris {len(sub)} != {len(ss)}")
    if not np.array_equal(sub.filename.to_numpy(), ss.filename.to_numpy()):
        errs.append("urutan filename != sample_submission")
    bad = set(sub.label.unique()) - set(CLASSES)
    if bad: errs.append(f"label aneh: {bad}")
    if len(sub.label.unique()) == 1:
        errs.append("RED FLAG: 100% satu kelas")
    else:
        vc = sub.label.value_counts(normalize=True)
        for c, exp in TRAIN_PROP.items():
            if abs(vc.get(c, 0.0) - exp) > 0.25:
                errs.append(f"proporsi {c}={vc.get(c,0):.3f} jauh dari train {exp}")
    if errs:
        print("[ASSERT GAGAL]"); [print("  -", e) for e in errs]; sys.exit(1)

    out = Path(args.out) if args.out else art / f"submission_{args.candidate}.csv"
    sub.to_csv(out, index=False)
    print(f"[ASSERT PASS] -> {out} ({len(sub)} baris)")
    print(f"distribusi: {sub.label.value_counts().to_dict()}")

    # log + (opsional) upload via Kaggle API
    if args.do_submit:
        rc, msg = _kaggle_submit(out, args.message)
        status = "SUBMITTED" if rc == 0 else f"FAILED({rc})"
        print(f"Kaggle submit: {status}\n{msg}")
        _log_submission(art, args.candidate, cv=args.basis_cv, filename=out.name,
                        note=args.message, status=status)
        sys.exit(rc)
    else:
        _log_submission(art, args.candidate, cv=args.basis_cv, filename=out.name,
                        note="assert pass (belum di-upload)", status="VALIDATED")
        print("CATATAN: upload/submit = jalankan ulang dengan --do-submit.")


# =====================================================================
# MAIN
# =====================================================================
def parse_stage():
    """Ambil argumen dari: (a) CLI argv, (b) variabel notebook STAGE/STAGE_ARGS.

    Di notebook (Kaggle/Colab), kalau file ini di-paste ke cell, __name__ != '__main__'
    style %run; argv berisi launcher notebook -> argparse error. Solusinya:
    set variabel STAGE = "smoke" dan STAGE_ARGS = "--name xxx" di cell terpisah,
    lalu panggil main() — atau pakai %run dengan argumen.
    """
    argv = sys.argv[1:]
    # di kernel notebook, argv milik launcher (mis. colab_kernel_launcher.py -f ...)
    # -> JANGAN pakai argv kecuali benar-benar berisi --stage
    if "--stage" not in argv:
        stage = globals().get("STAGE") or os.environ.get("DATASCAPE_STAGE")
        if not stage:
            raise SystemExit(
                "Argumen --stage hilang. Tiga cara pakai:\n"
                "  1) %run kaggle_all_in_one.py --stage smoke\n"
                "  2) set variabel di cell: STAGE = 'smoke'; STAGE_ARGS = '--name xxx'\n"
                "     lalu run cell berisi file ini ATAU panggil main()\n"
                "  3) terminal: python kaggle_all_in_one.py --stage train"
            )
        extra = globals().get("STAGE_ARGS") or ""
        argv = ["--stage", str(stage)] + (str(extra).split() if extra else [])
    return argv


def main():
    ap = argparse.ArgumentParser(description="DATASCAPE 2026 all-in-one (Kaggle)")
    ap.add_argument("--stage", required=True,
                    choices=["audit", "phash", "folds", "smoke", "train", "tta",
                             "ensemble", "submit", "all"],
                    help="audit=stage1, phash=stage2, folds=stage3, smoke/train=stage5, "
                         "tta=stage5b (hflip), ensemble=stage7, submit=stage9-build/assert")
    ap.add_argument("--name", default="resnet18_384")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--arch", default="resnet18",
                    help="backbone: resnet18/34/50 | efficientnet_b0/b2 | convnext_tiny/small | densenet121")
    ap.add_argument("--size", type=int, default=384, help="resolusi input (px)")
    ap.add_argument("--class-weight-cap", type=float, default=0.0,
                    help="cap bobot kelas N/(3c); >0 batas maks. 3.0 turunkan Lainnya 9.63->3.0")
    ap.add_argument("--strong-aug", action="store_true",
                    help="augmentasi drift-robust keras (JPEG q20-90, downscale 0.4-0.9, CJ 0.3)")
    ap.add_argument("--num-workers", type=int, default=2, help="Kaggle T4/P100: 2-4")
    ap.add_argument("--smoke", action="store_true", help="paksa smoke (implisit di --stage smoke)")
    ap.add_argument("--smoke-n", type=int, default=50)
    ap.add_argument("--cache", default=None,
                    help="pakai cache_768 (hasil stage4_cache.py) sebagai sumber gambar; "
                         "pad sudah diterapkan di cache, ids tetap filename asli")
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--scale-gate", type=float, default=0.02)
    ap.add_argument("--probs", default=None, help="stage submit: path test_probs npz")
    ap.add_argument("--out", default=None, help="stage submit: path output csv")
    ap.add_argument("--candidate", default="ensemble")
    ap.add_argument("--do-submit", action="store_true",
                    help="stage submit: upload ke Kaggle via CLI + log (J4)")
    ap.add_argument("--basis-cv", default="", help="stage submit: basis CV utk submission_log")
    ap.add_argument("--message", default="J4 submit", help="stage submit: pesan upload Kaggle")
    args = ap.parse_args(parse_stage())

    base_in, base_out = resolve_paths()
    art = base_out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    print(f"[paths] input={base_in} | artifacts={art}")
    print(f"[kaggle] is_kaggle={Path('/kaggle/input').is_dir()}")

    if args.stage == "smoke":
        args.smoke = True
        args.epochs = 2

    if args.stage == "all":
        stage1_audit(base_in, art)
        stage2_phash(base_in, art)
        stage3_folds(art)
        stage5_train(base_in, art, args)
        stage7_ensemble(base_in, art, args)
        stage9_submit(base_in, art, args)
        return

    if args.stage == "audit":
        stage1_audit(base_in, art)
    elif args.stage == "phash":
        stage2_phash(base_in, art)
    elif args.stage == "folds":
        stage3_folds(art)
    elif args.stage in ("smoke", "train"):
        # gerbang: folds.csv harus ada & hash-valid SEBELUM training (auto-rebuild
        # via SGKF vendored + blob BEKU — tahan beda versi sklearn di cloud)
        if not ensure_folds(base_in, art):
            sys.exit("[BLOKIR] folds.csv tidak valid — training ditolak.")
        stage5_train(base_in, art, args)
    elif args.stage == "tta":
        if not ensure_folds(base_in, art):
            sys.exit("[BLOKIR] folds.csv tidak valid — TTA ditolak.")
        stage5_tta(base_in, art, args)
    elif args.stage == "ensemble":
        verify_hash_lock(art, "folds.csv", fatal=False)
        stage7_ensemble(base_in, art, args)
    elif args.stage == "submit":
        stage9_submit(base_in, art, args)


if __name__ == "__main__":
    main()

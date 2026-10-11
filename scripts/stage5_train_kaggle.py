"""DATASCAPE 2026 — Stage 5: Fine-tune resnet18 @384px (J3 — Kamar Mesin).
Role (ORCHESTRATION.md / PROMPT_J3.md): J3 Validation & ML (DeepSeek V4 Pro) — Tahap 5.
Workflow: master_prompt_cnn_computer_vision.md Tahap 5 + 4.3/4.5 + Lampiran A.1–A.3.

KONTRAK OUTPUT (WAJIB — dipakai J4 / stage7_ensemble.py):
  artifacts/preds/<nama>_oof.npz    keys: ids (str[n_train]), probs (float32[n_train,3])
  artifacts/preds/<nama>_test.npz   keys: ids (str[n_test]),  probs (float32[n_test,3])
  - ids oof  = URUTAN PERSIS folds.csv (kolom image_id)
  - ids test = URUTAN PERSIS sample_submission.csv (kolom filename)
  - probs = softmax 3 kelas: Aman=0, Rusak=1, Lainnya=2
  - test = FOLD-MEAN (rata-rata probabilitas 5 model per-fold) — keputusan D10.

Desain (keputusan beku di decision_log.md):
  - resnet18 ImageNet-pretrained; head Dropout(0.3)+Linear(3); 384px.
  - AdamW backbone 1e-4 / head 1e-3, wd 1e-2; warmup 2 + cosine; AMP fp16; grad-clip 1.0.
  - CE class-weighted N/(3*count) + label smoothing 0.1 (D12) — TANPA oversampling.
  - Augmentasi (D8 — serang drift kompresi/resolusi): PAD ke persegi (bukan squash, D6),
    RandomResizedCrop(0.8-1.0), HFlip, Affine ±10°, ColorJitter 0.2,
    JPEG-compression aug (quality 30-95) + random-downscale (0.5-0.9), RandomErasing p=0.25.
  - Eval transform DETERMINISTIK identik val & test (PAD -> resize 384) — test-time parity.
  - CEK COLLAPSE tiap epoch (distribusi prediksi val; 100% 1 kelas = rusak).
  - Simpan bobot TERBAIK per fold (val macro-F1, bukan epoch terakhir).

Portable: auto-detect path lokal (Windows) atau /kaggle/input (Kaggle GPU).
Pakai di Kaggle notebook (GPU T4/P100):
  %run stage5_train_kaggle.py --name resnet18_384          # full 5 fold
  %run stage5_train_kaggle.py --smoke                      # 1 split, ~50 img, 2 epoch
  %run stage5_train_kaggle.py --smoke --smoke-n 24 --epochs 1 --batch-size 8   # verifikasi CPU
"""
import argparse
import io
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------------------- konfigurasi default -----------------------------
CLASSES = ["Aman", "Rusak", "Lainnya"]          # Aman=0, Rusak=1, Lainnya=2
SEED = 42
IMG_SIZE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_CAP = 768           # sisi terpanjang di-cap sebelum PAD (batasi memori 8000px)


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


def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------- metrik -----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(CLASSES)
    out = 0.0
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out += 2 * p * r / (p + r) if (p + r) else 0.0
    return out / n


# ----------------------------- transform -----------------------------
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
    """Wraper callable picklable (wajib utk DataLoader workers di Windows/spawn)."""
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


def build_transforms():
    train_tf = transforms.Compose([
        PadToSquare(PAD_CAP),
        RandomDownscale(p=0.5, scale=(0.5, 0.9)),   # simulasi low-res SEBELUM resize final
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        RandomJPEGCompression(p=0.5, quality=(30, 95)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value=0.0),
    ])
    eval_tf = transforms.Compose([          # DETERMINISTIK, identik val & test
        PadToSquare(PAD_CAP),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ----------------------------- dataset -----------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")   # convert menangani RGBA/P
        img = self.transform(img)
        if self.labels is None:
            return img
        return img, self.labels[i]


# ----------------------------- model -----------------------------
def build_model(arch="resnet18", num_classes=3, pretrained=True, dropout=0.3):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    feat = m.fc.in_features
    m.fc = nn.Identity()
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))
    model = nn.Sequential(m, head)
    return model, feat


# ----------------------------- train / predict -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, grad_clip):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, use_amp):
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


# ----------------------------- util logging -----------------------------
def append_metrics(base, name, cv_mean, cv_std, oof_best, note):
    p = Path(base) / "artifacts" / "metrics_summary.csv"
    cols = ["experiment", "stage", "cv_mean", "cv_std", "oof_best",
            "lb_public", "lb_private", "note", "date"]
    row = {"experiment": name, "stage": 5,
           "cv_mean": round(float(cv_mean), 4), "cv_std": round(float(cv_std), 4),
           "oof_best": round(float(oof_best), 4), "lb_public": "", "lb_private": "",
           "note": note, "date": datetime.now().strftime("%Y-%m-%d")}
    if p.exists():
        df = pd.read_csv(p)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=cols)
    df.to_csv(p, index=False)


# ----------------------------- main -----------------------------
def main(args):
    t0 = time.time()
    set_seed(SEED)
    base = _find_base()
    art = _writable_art(base)
    (art / "preds").mkdir(exist_ok=True)
    (art / "ckpt").mkdir(exist_ok=True)

    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    device = torch.device("cuda" if cuda_ok else "cpu")
    use_amp = device.type == "cuda"
    n_workers = args.num_workers
    print(f"[stage5] device={device} | amp={use_amp} | workers={n_workers}")
    if device.type != "cuda":
        print("[stage5] WARNING: berjalan di CPU — untuk produksi jalankan di Kaggle GPU.")

    # ---- load folds + test ----
    folds = pd.read_csv(art / "folds.csv", dtype={"image_id": str})
    folds["path"] = [str(base / "Train_Set" / lbl / fn)
                     for lbl, fn in zip(folds["label"], folds["image_id"])]
    y_all = folds["target"].to_numpy(dtype=np.int64)

    ss = pd.read_csv(base / "sample_submission.csv")
    test_paths = [str(base / "Test_Set" / fn) for fn in ss["filename"]]
    test_ids = ss["filename"].to_numpy(dtype=str)

    if args.smoke:
        # smoke: prediksi test cukup subset kecil (verifikasi format saja, hemat CPU)
        test_paths = test_paths[:args.smoke_n]
        test_ids = test_ids[:args.smoke_n]
        if args.name.endswith("_smoke"):
            pass
        else:
            args.name = f"{args.name}_smoke"

    # ---- split definition ----
    if args.smoke:
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
        tr_idx = np.array(tr_idx); va_idx = np.array(va_idx)
        split_list = [(tr_idx, va_idx)]
        folds = sub.reset_index(drop=True)
        y_all = sub["target"].to_numpy(dtype=np.int64)
        print(f"[smoke] subset n={len(sub)} (train {len(tr_idx)} / val {len(va_idx)})")
    else:
        split_list = []
        for f in range(args.folds):
            va = np.where(folds["fold"].to_numpy() == f)[0]
            tr = np.where(folds["fold"].to_numpy() != f)[0]
            split_list.append((tr, va))

    # ---- class weights (N/(3*count)) ----
    counts = np.bincount(y_all, minlength=3)
    weights = len(y_all) / (3.0 * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[stage5] class counts={counts.tolist()} | weights={weights.round(4).tolist()}")

    train_tf, eval_tf = build_transforms()
    test_ds = ImageDataset(test_paths, np.zeros(len(test_paths), dtype=np.int64), eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=n_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=(n_workers > 0))

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
                               num_workers=n_workers, pin_memory=(device.type == "cuda"),
                               persistent_workers=(n_workers > 0))
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=n_workers, pin_memory=(device.type == "cuda"),
                               persistent_workers=(n_workers > 0))

        model, _ = build_model(arch=args.arch, num_classes=3, pretrained=True, dropout=args.dropout)
        model.to(device)
        backbone = model[0]
        head = model[1]
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
        best_val_probs = None

        for ep in range(args.epochs):
            loss = train_one_epoch(model, tr_loader, optimizer, criterion, scaler,
                                   device, use_amp, args.grad_clip)
            scheduler.step()
            vp = predict(model, va_loader, device, use_amp)
            dist, f1, collapsed = collapse_report(vp, y_all[va_idx])
            flag = "  <-- COLLAPSE!" if collapsed else ""
            print(f"  ep {ep}: loss={loss:.4f} val_macroF1={f1:.4f} dist={dist}{flag}")
            if collapsed:
                print("  [WARN] distribusi val 100% satu kelas — model rusak; periksa LR/label/transform.")
            if f1 > best_f1:
                best_f1 = f1
                best_val_probs = vp
                torch.save(model.state_dict(), best_path)

        # load bobot terbaik -> prediksi OOF + test
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        best_val_probs = predict(model, va_loader, device, use_amp)
        _, best_f1, _ = collapse_report(best_val_probs, y_all[va_idx])
        oof_probs[va_idx] = best_val_probs
        fold_f1s.append(best_f1)
        print(f"  fold {fi} BEST val macro-F1 = {best_f1:.4f} (ckpt: {best_path.name})")

        tp = predict(model, test_loader, device, use_amp)
        test_sum = tp if test_sum is None else test_sum + tp

    test_probs = test_sum / len(split_list)
    fold_f1s = np.array(fold_f1s)
    cv_mean, cv_std = float(fold_f1s.mean()), float(fold_f1s.std())
    oof_pred = oof_probs.argmax(1)
    oof_best = macro_f1(y_all, oof_pred)
    print(f"\n===== HASIL =====\n  fold F1: {fold_f1s.round(4).tolist()}\n"
          f"  CV macro-F1 = {cv_mean:.4f} ± {cv_std:.4f} | OOF = {oof_best:.4f}")

    # ---- simpan KONTRAK J4 ----
    np.savez_compressed(art / "preds" / f"{args.name}_oof.npz", ids=oof_ids, probs=oof_probs.astype(np.float32))
    np.savez_compressed(art / "preds" / f"{args.name}_test.npz", ids=test_ids, probs=test_probs.astype(np.float32))
    print(f"[save] {args.name}_oof.npz  shape={oof_probs.shape} ids[0]={oof_ids[0]}")
    print(f"[save] {args.name}_test.npz shape={test_probs.shape} ids[0]={test_ids[0]}")

    # sanity distribusi prediksi OOF
    od = {CLASSES[c]: int((oof_pred == c).sum()) for c in range(len(CLASSES))}
    print(f"[sanity] distribusi prediksi OOF: {od} (train ~ 72/24.6/3.5)")

    if args.smoke:
        print(f"[save] SKIP metrics_summary (smoke, bukan eksperimen sah)")
    else:
        append_metrics(base, args.name, cv_mean, cv_std, oof_best,
                       f"resnet18@{IMG_SIZE}; aug=pad+RRC+hflip+aff10+cj0.2+jpeg+downscale+erase; "
                       f"classweight+LS0.1; fold-mean test; runtime {time.time()-t0:.0f}s")
    print(f"\nTOTAL RUNTIME {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="resnet18_384")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=6,
                    help="DataLoader workers; 6 utk 16-thread lokal, 2 utk Kaggle")
    ap.add_argument("--smoke", action="store_true", help="1 split, ~50 img, 2 epoch")
    ap.add_argument("--smoke-n", type=int, default=50)
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 2
    main(a)

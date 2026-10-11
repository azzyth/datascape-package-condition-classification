# -*- coding: utf-8 -*-
"""TRACK A — DINOv2 frozen-embedding probe (J1/J3 gate sebelum Track B fine-tune).

Ekstraksi embedding DINOv2 (frozen, tanpa training) untuk train+test, lalu:
  1. Linear probe (LogReg) OOF di fold BEKU (artifacts/folds.csv)
  2. kNN cosine (group-aware lewat fold) OOF
GATE: kalau probe OOF >= ~0.80 -> DINOv2 layak di-fine-tune (Track B).

Jalan LOKAL (RTX 5060). Pakai cache_768 (gambar sudah 768px JPEG bersih).
Output: artifacts/embeddings/dinov2_{train,test}.npz + probe report ke stdout.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:/Users/PONGO AMD/Documents/codingstuff/KAGGLE_COMPETITIONS/datascape")
ART = BASE / "artifacts"
CACHE = BASE / "cache_768"
EMB_DIR = ART / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Aman", "Rusak", "Lainnya"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 42

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[trackA] device={device} torch={torch.__version__}")


class ImgDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
        # DINOv2 (timm vit_base_patch14_dinov2) default: 518px, IN mean/std.
        # Kita resize ke 518 (kelipatan 14) — patch14 butuh kelipatan 14.
        self.tf = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


@torch.no_grad()
def extract(model, loader):
    model.eval()
    feats = []
    for x in loader:
        x = x.to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            f = model(x)  # num_classes=0 -> embedding
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy()


def main():
    import timm
    folds = pd.read_csv(ART / "folds.csv", dtype={"image_id": str})
    y = folds["label"].map(CLS2IDX).to_numpy()
    fold_of = folds["fold"].to_numpy()
    train_paths = [str(CACHE / "Train_Set" / lbl / (Path(fn).stem + ".jpg"))
                   for lbl, fn in zip(folds["label"], folds["image_id"])]

    ss = pd.read_csv(BASE / "sample_submission.csv")
    test_paths = [str(CACHE / "Test_Set" / (Path(fn).stem + ".jpg")) for fn in ss["filename"]]
    test_ids = ss["filename"].to_numpy(dtype=str)

    print(f"[trackA] train={len(train_paths)} test={len(test_paths)}")

    emb_tr = EMB_DIR / "dinov2_train.npz"
    emb_te = EMB_DIR / "dinov2_test.npz"
    if emb_tr.is_file() and emb_te.is_file():
        print("[trackA] embeddings ada -> load dari disk (skip ekstraksi)")
        zt = np.load(emb_tr, allow_pickle=False)
        Xtr = zt["feats"].astype(np.float64)
        assert np.array_equal(zt["ids"].astype(str), folds["image_id"].to_numpy().astype(str))
        Xte = np.load(emb_te, allow_pickle=False)["feats"].astype(np.float64)
    else:
        print("[trackA] load DINOv2 (timm vit_base_patch14_reg4_dinov2.lvd142m)...")
        model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m",
                                  pretrained=True, num_classes=0)
        model.to(device)
        feat_dim = model.num_features
        print(f"[trackA] feat_dim={feat_dim}")

        bs = 32
        nw = 6
        tr_loader = DataLoader(ImgDS(train_paths), batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True, persistent_workers=True)
        te_loader = DataLoader(ImgDS(test_paths), batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True, persistent_workers=True)

        t0 = time.time()
        Xtr = extract(model, tr_loader)
        print(f"[trackA] train embed {Xtr.shape} in {time.time()-t0:.0f}s")
        Xte = extract(model, te_loader)
        print(f"[trackA] test embed {Xte.shape}")

        np.savez_compressed(EMB_DIR / "dinov2_train.npz",
                            ids=folds["image_id"].to_numpy(dtype=str), feats=Xtr.astype(np.float32),
                            y=y, fold=fold_of)
        np.savez_compressed(EMB_DIR / "dinov2_test.npz",
                            ids=test_ids, feats=Xte.astype(np.float32))
        print(f"[trackA] saved -> {EMB_DIR}")

    # ---- 1. Linear probe (LogReg) OOF di fold beku ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    oof_lr = np.zeros((len(y), 3), dtype=np.float64)
    for f in range(5):
        tr = fold_of != f
        va = fold_of == f
        sc = StandardScaler().fit(Xtr[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 solver="lbfgs")
        clf.fit(sc.transform(Xtr[tr]), y[tr])
        oof_lr[va] = clf.predict_proba(sc.transform(Xtr[va]))
        f1f = f1_score(y[va], oof_lr[va].argmax(1), average="macro")
        print(f"  [LR] fold {f}: macro-F1={f1f:.4f}")
    oof_lr_f1 = f1_score(y, oof_lr.argmax(1), average="macro")
    print(f"[LR] OOF macro-F1 = {oof_lr_f1:.5f}")

    # ---- 2. kNN cosine OOF (group-aware via fold) ----
    from sklearn.preprocessing import normalize
    Xn = normalize(Xtr)  # cosine -> dot pada vektor ternormalisasi
    oof_knn = np.zeros((len(y), 3), dtype=np.float64)
    K = 15
    for f in range(5):
        tr = np.where(fold_of != f)[0]
        va = np.where(fold_of == f)[0]
        A = Xn[tr]              # (ntr, d)
        ytr = y[tr]
        for i in va:
            sim = A @ Xn[i]     # cosine sim ke semua train
            top = np.argpartition(-sim, K)[:K]
            # bobot = sim (clip >=0), vote per kelas
            w = np.clip(sim[top], 0, None)
            votes = np.zeros(3)
            for c in range(3):
                votes[c] = w[ytr[top] == c].sum()
            if votes.sum() == 0:
                votes = np.bincount(ytr, minlength=3)  # fallback prior
            oof_knn[i] = votes / votes.sum()
        f1f = f1_score(y[va], oof_knn[va].argmax(1), average="macro")
        print(f"  [kNN k={K}] fold {f}: macro-F1={f1f:.4f}")
    oof_knn_f1 = f1_score(y, oof_knn.argmax(1), average="macro")
    print(f"[kNN k={K}] OOF macro-F1 = {oof_knn_f1:.5f}")

    # ---- GATE ----
    print("\n===== TRACK A GATE =====")
    print(f"  DINOv2 frozen linear-probe OOF = {oof_lr_f1:.5f}")
    print(f"  DINOv2 kNN(k={K}) OOF          = {oof_knn_f1:.5f}")
    print(f"  convnext_tiny fine-tune OOF    = 0.84641 (referensi)")
    verdict = "LAYAK fine-tune (Track B)" if max(oof_lr_f1, oof_knn_f1) >= 0.80 else "LEMAH — pertimbangkan skip"
    print(f"  VERDICT: {verdict}")

    # simpan probe probs utk kemungkinan ensemble
    np.savez_compressed(EMB_DIR / "dinov2_lr_oof.npz",
                        ids=folds["image_id"].to_numpy(dtype=str), probs=oof_lr.astype(np.float32))
    np.savez_compressed(EMB_DIR / "dinov2_knn_oof.npz",
                        ids=folds["image_id"].to_numpy(dtype=str), probs=oof_knn.astype(np.float32))
    print(f"[trackA] probe probs saved -> {EMB_DIR}")


if __name__ == "__main__":
    main()

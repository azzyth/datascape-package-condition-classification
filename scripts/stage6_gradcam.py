# -*- coding: utf-8 -*-
"""DATASCAPE 2026 — Stage 6b: Grad-CAM (interpretability evidence, Tahap 6 / Lampiran A.7).

Tujuan: bukti visual bahwa model melihat AREA PAKET (bukan watermark/bingkai/artefak).
Model: convnext_tiny_512 (ckpt lokal 5 fold tersedia; backbone sama dgn pemenang base).
Output: artifacts/gradcam_grid.jpg (montage asli | heatmap-overlay per contoh, label + prediksi).

Metode: manual gradient-CAM via forward+backward hooks pada blok konvolusi terakhir
(model[0].features[-1] untuk torchvision convnext). CAM = ReLU(sum_k w_k * A_k),
w_k = mean_spatial(grad skor kelas target thdp A_k). Tanpa dependency pytorch-grad-cam.
"""
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(r"C:/Users/PONGO AMD/Documents/codingstuff/KAGGLE_COMPETITIONS/datascape")
TRAIN, ART = BASE / "Train_Set", BASE / "artifacts"
CKPT = ART / "ckpt"
CLASSES = ["Aman", "Rusak", "Lainnya"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_CAP = 768
SIZE = 512          # convnext_tiny_512
MODEL_NAME = "convnext_tiny_512"
FOLD_FOR_CAM = 0    # satu fold ckpt cukup untuk visual diagnostik
N_PER_CLASS = 3     # contoh per kelas (worst-error + correct)
SEED = 42

import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[gradcam] device={device} torch={torch.__version__}")


def pad_to_square(img, cap=PAD_CAP):
    img = img.convert("RGB"); w, h = img.size
    s = cap / max(w, h) if max(w, h) > cap else 1.0
    if s < 1.0:
        img = img.resize((max(1, round(w*s)), max(1, round(h*s))), Image.BILINEAR); w, h = img.size
    side = max(w, h); canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side-w)//2, (side-h)//2)); return canvas


eval_tf = transforms.Compose([
    transforms.Lambda(lambda im: pad_to_square(im, PAD_CAP)),
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])


def build_model(num_classes=3):
    m = models.convnext_tiny(weights=None)  # bobot di-load dari ckpt
    feat = m.classifier[2].in_features
    m.classifier[2] = nn.Identity()
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, num_classes))
    return nn.Sequential(m, head)


def load_fold_model(fold):
    model = build_model()
    ck = CKPT / f"{MODEL_NAME}_fold{fold}_best.pt"
    sd = torch.load(ck, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ---------- Grad-CAM (manual, hooks pada features[-1]) ----------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.act = None
        self.grad = None
        target = model[0].features[-1]  # blok konvolusi terakhir convnext
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.act = o.detach()

    def _bwd(self, m, gi, go):
        self.grad = go[0].detach()

    def __call__(self, x, target_class):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)  # (1,3)
        score = logit[0, target_class]
        score.backward()
        # convnext features output: (1, C, H, W) -> tapi convnext pakai channels_last, aman
        A = self.act[0]                     # (C,H,W)
        g = self.grad[0]                    # (C,H,W)
        w = g.mean(dim=(1, 2))              # (C,)
        cam = torch.relu((w[:, None, None] * A).sum(0))  # (H,W)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), logit.detach().cpu().numpy()[0]


def overlay_cam(pil_img, cam, alpha=0.5):
    """pil_img: RGB 512. cam: (H,W) 0..1. -> overlay jet-ish."""
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    base = np.asarray(pil_img.resize((SIZE, SIZE))).astype(np.float32) / 255.0
    heat = cmap(cam)[..., :3]  # (H,W,3) 0..1
    heat_img = np.asarray(Image.fromarray((heat*255).astype(np.uint8)).resize((SIZE, SIZE))).astype(np.float32)/255.0
    out = (1-alpha)*base + alpha*heat_img
    return Image.fromarray((np.clip(out, 0, 1)*255).astype(np.uint8))


def main():
    folds = pd.read_csv(ART / "folds.csv", dtype={"image_id": str})
    oof = np.load(ART / "preds" / f"{MODEL_NAME}_oof.npz")
    assert np.array_equal(oof["ids"].astype(str), folds["image_id"].to_numpy().astype(str))
    probs = oof["probs"].astype(np.float64)
    y = folds["target"].to_numpy()
    pred = probs.argmax(1); conf = probs.max(1)

    folds["path"] = [str(TRAIN / lbl / fn) for lbl, fn in zip(folds["label"], folds["image_id"])]

    # pilih contoh: per kelas -> worst-error (salah, conf tinggi) + correct (benar, conf tinggi)
    rng = np.random.default_rng(SEED)
    picks = []  # (idx, tag)
    for c in range(3):
        mask = (y == c)
        wrong = np.where(mask & (pred != c))[0]
        right = np.where(mask & (pred == c))[0]
        wrong = wrong[np.argsort(-conf[wrong])][:N_PER_CLASS]      # paling salah & paling yakin
        right = right[np.argsort(-conf[right])][:N_PER_CLASS]      # paling benar & yakin
        for i in wrong: picks.append((i, f"TRUE={CLASSES[y[i]]} PRED={CLASSES[pred[i]]} (salah)"))
        for i in right: picks.append((i, f"TRUE={CLASSES[y[i]]} PRED={CLASSES[pred[i]]} (benar)"))

    model = load_fold_model(FOLD_FOR_CAM)
    cammer = GradCAM(model)
    print(f"[gradcam] model fold{FOLD_FOR_CAM} loaded; {len(picks)} contoh")

    tiles = []
    for idx, tag in picks:
        path = folds["path"].iloc[idx]
        pil = Image.open(path).convert("RGB")
        x = eval_tf(pil).unsqueeze(0).to(device)
        tc = int(pred[idx])  # CAM terhadap kelas yang DIPREDIKSI (jelaskan keputusan model)
        cam, logit = cammer(x, tc)
        base_img = pil.resize((SIZE, SIZE))
        ov = overlay_cam(pil, cam)
        tiles.append((base_img, ov, tag))
        print(f"  [{CLASSES[y[idx]]}->{CLASSES[tc]}] conf={conf[idx]:.2f} cam_max={cam.max():.2f}  {folds['image_id'].iloc[idx]}")

    # montage: tiap contoh = (asli | overlay) side-by-side
    cols = 2
    rows = len(tiles)
    Wtile, Htile = SIZE, SIZE
    head = 26
    canvas = Image.new("RGB", (cols*(Wtile+8), rows*(Htile+head)), "white")
    dr = ImageDraw.Draw(canvas)
    for r, (b, o, tag) in enumerate(tiles):
        y0 = r*(Htile+head)
        dr.text((6, y0+5), tag[:80], fill="black")
        canvas.paste(b, (0, y0+head))
        canvas.paste(o, (Wtile+8, y0+head))
    out = ART / "gradcam_grid.jpg"
    canvas.save(out, quality=90)
    print(f"\n[gradcam] tersimpan: {out}  ({len(tiles)} contoh, kolom: asli | Grad-CAM overlay)")
    print("[gradcam] BACA MAKSUD: area merah/hangat = region yang dipakai model utk prediksi.")


if __name__ == "__main__":
    main()

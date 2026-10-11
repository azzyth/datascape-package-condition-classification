"""DATASCAPE 2026 — Stage 2c: EDA Visual + Adversarial Validation (Jendela 2).
Workflow: master_prompt_cnn_computer_vision.md Tahap 2 (2.0, 2.1, 2.2).
Output (artifacts/):
  eda_grid_{Aman,Rusak,Lainnya}.jpg   4x5 grid per kelas (verifikasi mata user)
  eda_difficult.jpg                   gambar sulit: gelap/terang/kecil/aspek ekstrem
  eda_size_dist.png, eda_intensity.png, eda_adversarial.png
  stage2c_eda_report.md               angka + keputusan desain
"""
import os, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    raise RuntimeError("Dataset base tidak ditemukan.")



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
TRAIN, TEST = BASE / "Train_Set", BASE / "Test_Set"
ART = _writable_art(BASE)
CLASSES = ["Aman", "Rusak", "Lainnya"]
SEED = 42
rng = np.random.default_rng(SEED)
t0 = time.time()

# ---------- load cheap features (stage 1) ----------
tr = pd.read_csv(ART / "cheap_features_train.csv")
te = pd.read_csv(ART / "cheap_features_test.csv")
tr["bpp"] = tr.bytes / (tr.w * tr.h)
te["bpp"] = te.bytes / (te.w * te.h)
print(f"features: train={len(tr)} test={len(te)}")

# ============ 1) distribusi kelas ============
print("\n=== DISTRIBUSI KELAS (train) ===")
dist = tr.label.value_counts()
for c in CLASSES:
    print(f"  {c:8s}: {dist[c]:5d} ({100*dist[c]/len(tr):.1f}%)")

# ============ 2) distribusi ukuran & aspect ============
def pctl(s, qs=(5, 25, 50, 75, 95)):
    return {f"p{q}": round(float(np.percentile(s, q)), 2) for q in qs}

print("\n=== UKURAN (train | test) ===")
for col in ["w", "h", "aspect", "bpp"]:
    print(f"  {col:7s}: train {pctl(tr[col])} | test {pctl(te[col])}")

pct_out = float(((tr.aspect < 0.75) | (tr.aspect > 1.33)).mean())
print(f"  train aspect di luar 0.75-1.33: {100*pct_out:.1f}% -> {'PAD (bukan squash)' if pct_out > 0.05 else 'squish aman'}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist([tr.w, te.w], bins=60, label=["train", "test"], range=(0, 2000))
axes[0].set_title("width"); axes[0].legend()
axes[1].hist([tr.h, te.h], bins=60, label=["train", "test"], range=(0, 2000))
axes[1].set_title("height")
axes[2].scatter(tr.w.sample(3000, random_state=SEED), tr.h.sample(3000, random_state=SEED),
                s=3, alpha=0.25, c=[{"Aman": "tab:blue", "Rusak": "tab:red", "Lainnya": "tab:green"}[l] for l in tr.label.sample(3000, random_state=SEED)])
axes[2].set_title("train w vs h (biru=Aman merah=Rusak hijau=Lainnya)")
axes[2].set_xlabel("w"); axes[2].set_ylabel("h")
plt.tight_layout(); plt.savefig(ART / "eda_size_dist.png", dpi=110); plt.close()
print(f"  plot: eda_size_dist.png")

# ============ 3) statistik pixel per kelas (sample decode 64px) ============
S = 64
def decode_stats(row):
    try:
        p = TRAIN / row.label / row.filename
        a = np.asarray(Image.open(p).convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.
        return a.mean(), a.reshape(-1, 3), os.path.getsize(p)
    except Exception:
        return None

def _sample_by_class(df, n_per_class, seed=SEED):
    parts = [d.sample(min(n_per_class, len(d)), random_state=seed)
             for _, d in df.groupby("label")]
    return pd.concat(parts)

samp = _sample_by_class(tr, 700)
print(f"\n=== STATISTIK PIXEL per kelas (sample {len(samp)} gambar @64px) ===")
chan_stats, hists = {}, {}
for c in CLASSES:
    sub = samp[samp.label == c]
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = [r for r in ex.map(decode_stats, [r for _, r in sub.iterrows()]) if r]
    pix = np.concatenate([r[1] for r in res])  # n x 3
    luma = [r[0] for r in res]
    chan_stats[c] = {"mean": pix.mean(0).round(3), "std": pix.std(0).round(3),
                     "luma_mean": round(float(np.mean(luma)), 3)}
    hists[c] = pix.mean(1)  # intensitas per gambar-pixel avg
    print(f"  {c:8s}: mean={pix.mean(0).round(3)} std={pix.std(0).round(3)} luma={np.mean(luma):.3f}")
imnet_mean, imnet_std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
allmean = np.mean([v["mean"] for v in chan_stats.values()], axis=0)
allstd = np.mean([v["std"] for v in chan_stats.values()], axis=0)
print(f"  dataset mean={allmean.round(3)} std={allstd.round(3)}")
print(f"  ImageNet mean={imnet_mean} std={imnet_std}")
print(f"  -> normalisasi: {'dataset stats layak (beda jauh dari ImageNet)' if np.abs(allmean-imnet_mean).max()>0.05 else 'ImageNet OK'}")

plt.figure(figsize=(7, 4))
for c in CLASSES:
    plt.hist(hists[c], bins=60, alpha=0.5, label=c, density=True, range=(0, 1))
plt.title("distribusi intensitas per kelas"); plt.legend()
plt.tight_layout(); plt.savefig(ART / "eda_intensity.png", dpi=110); plt.close()
print("  plot: eda_intensity.png")

# ============ 4) contoh gambar sulit ============
def montage(paths_titles, out, cols=4, size=200):
    rows = int(np.ceil(len(paths_titles) / cols))
    H = size + 34
    canvas = Image.new("RGB", (cols * (size + 8), rows * H), "white")
    dr = ImageDraw.Draw(canvas)
    for i, (p, title) in enumerate(paths_titles):
        x, y = (i % cols) * (size + 8), (i // cols) * H
        dr.text((x + 4, y + 6), title[:60], fill="black")
        try:
            im = Image.open(p).convert("RGB"); im.thumbnail((size, size))
            canvas.paste(im, (x + 4, y + 28))
        except Exception as e:
            dr.text((x + 4, y + 28), f"ERR {e}", fill="red")
    canvas.save(out, quality=88)
    print(f"  montage: {out.name}")

def P(label, fname): return TRAIN / label / fname
tr["luma"] = (tr.r + tr.g + tr.b) / 3
tr["pix"] = tr.w * tr.h
difficult = []
for name, sub in [("gelap", tr.nsmallest(3, "luma")), ("terang", tr.nlargest(3, "luma")),
                  ("terkecil", tr.nsmallest(3, "pix")), ("aspect-extreme", tr.nlargest(3, "aspect")),
                  ("bpp-min", tr.nsmallest(3, "bpp"))]:
    for _, r in sub.iterrows():
        difficult.append((P(r.label, r.filename), f"{name} | {r.label} | {r.filename}"))
montage(difficult, ART / "eda_difficult.jpg", cols=5)

# ============ 5) grid per kelas (4x5) untuk inspeksi mata user ============
for c in CLASSES:
    sub = tr[tr.label == c].sample(20, random_state=SEED)
    montage([(P(c, r.filename), f"{r.filename} | {int(r.w)}x{int(r.h)}") for _, r in sub.iterrows()],
            ART / f"eda_grid_{c}.jpg", cols=5, size=180)

# ============ 6) EXIF audit (jejak sumber/scrape) ============
def exif_one(row):
    try:
        with Image.open(TRAIN / row.label / row.filename) as im:
            e = im.getexif()
            sw = e.get(305, "") or ""  # Software
            mk = e.get(271, "") or ""  # Make
            return f"{sw}|{mk}" if (sw or mk) else ""
    except Exception:
        return "ERR"
sub = _sample_by_class(tr, 500)
with ThreadPoolExecutor(max_workers=16) as ex:
    exif_res = list(ex.map(exif_one, [r for _, r in sub.iterrows()]))
sub = sub.assign(exif=exif_res)
has_exif = sub[sub.exif != ""]
print(f"\n=== EXIF (sample {len(sub)}) ===")
print(f"  punya EXIF/software: {len(has_exif)} ({100*len(has_exif)/len(sub):.1f}%)")
if len(has_exif):
    vc = has_exif.exif.value_counts().head(8)
    for k, v in vc.items():
        print(f"  {v:4d}x {k[:70]}")

# ============ 7) ADVERSARIAL VALIDATION (drift train vs test) ============
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

FEATS = ["r", "g", "b", "sr", "sg", "sb", "w", "h", "aspect", "bytes", "bpp"]
X = pd.concat([tr[FEATS], te[FEATS]]).values
y = np.concatenate([np.zeros(len(tr)), np.ones(len(te))])
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
aucs, imps = [], []
for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
    m = RandomForestClassifier(300, n_jobs=-1, random_state=SEED, class_weight="balanced")
    m.fit(X[i_tr], y[i_tr])
    p = m.predict_proba(X[i_va])[:, 1]
    aucs.append(roc_auc_score(y[i_va], p))
    imps.append(m.feature_importances_)
print(f"\n=== ADVERSARIAL VALIDATION (train vs test) ===")
print(f"  RF AUC per fold: {[round(a,3) for a in aucs]}  mean={np.mean(aucs):.4f}")
imp = pd.Series(np.mean(imps, 0), index=FEATS).sort_values(ascending=False)
print("  feature importance:")
for k, v in imp.items():
    print(f"    {k:7s} {v:.3f}")
verdict = ("SE-DISTRIBUSI (AUC<0.6)" if np.mean(aucs) < 0.6 else
           "DRIFT SEDANG (0.6-0.8)" if np.mean(aucs) < 0.8 else "DRIFT KUAT (>0.8)")
print(f"  verdict: {verdict}")
# univariate AUC fitur teratas
from sklearn.metrics import roc_auc_score as auc1
uni = {f: round(auc1(y, X[:, i]), 3) for i, f in enumerate(FEATS)}
print(f"  univariate AUC: { {k:v for k,v in sorted(uni.items(), key=lambda kv:-abs(kv[1]-0.5))[:5]} }")

plt.figure(figsize=(7, 4))
plt.bar(imp.index, imp.values)
plt.title(f"adversarial validation: fitur pembeda train vs test (AUC={np.mean(aucs):.3f})")
plt.tight_layout(); plt.savefig(ART / "eda_adversarial.png", dpi=110); plt.close()
print("  plot: eda_adversarial.png")

# ============ 8) laporan ============
rep = f"""# Stage 2c — EDA Visual + Adversarial Validation (J2)
## Kelas
{dist.to_dict()} (21:7:1) — class-weighted CE / oversampling Lainnya wajib.

## Ukuran & aspect
- train: w p50={pctl(tr.w)['p50']} h p50={pctl(tr.h)['p50']} aspect p5-p95=[{np.percentile(tr.aspect,5):.2f},{np.percentile(tr.aspect,95):.2f}]
- test : w p50={pctl(te.w)['p50']} h p50={pctl(te.h)['p50']} aspect p5-p95=[{np.percentile(te.aspect,5):.2f},{np.percentile(te.aspect,95):.2f}]
- {100*pct_out:.1f}% train di luar aspect 0.75-1.33 -> strategi resize: **PAD ke persegi** (jangan squash)
- bpp train p50={pctl(tr.bpp)['p50']:.2f} vs test p50={pctl(te.bpp)['p50']:.2f}

## Statistik pixel (sample @64px)
- per kelas: { {k: chan_stats[k] for k in CLASSES} }
- dataset mean={allmean.round(3).tolist()} std={allstd.round(3).tolist()}
- keputusan normalisasi: pakai ImageNet dulu (kompabilitas pretrained), catat dataset stats sebagai alternatif

## EXIF
- {len(has_exif)}/{len(sub)} sample punya EXIF software/make; detail di output console
  (indikasi scrape multi-sumber; watermark stok foto terlihat di montase)

## Adversarial validation
- RF AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} -> **{verdict}**
- fitur pembeda teratas: {imp.head(3).to_dict()}
- univariate AUC (5 teratas): { {k:v for k,v in sorted(uni.items(), key=lambda kv:-abs(kv[1]-0.5))[:5]} }
- implikasi: lihat laporan utuh untuk keputusan augmentasi (serang fitur drift: resolusi/kompresi)

## Artefak
- eda_grid_{{Aman,Rusak,Lainnya}}.jpg — WAJIB dibaca mata user (cari watermark/bingkai/artefak per kelas)
- eda_difficult.jpg — kasus gelap/terang/kecil/aspek ekstrem/bpp rendah
- eda_size_dist.png, eda_intensity.png, eda_adversarial.png

Runtime {time.time()-t0:.1f}s
"""
(ART / "stage2c_eda_report.md").write_text(rep, encoding="utf-8")
print(f"\nreport: stage2c_eda_report.md | runtime {time.time()-t0:.1f}s")

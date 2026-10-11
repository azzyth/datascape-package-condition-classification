"""DATASCAPE 2026 — Stage 2b: verifikasi numerik pasangan yang di-flag pHash.
Hitung pixel MSE (resize 128x128, grayscale) untuk:
  1. pasangan train<->test d<=6 (sampling)
  2. pasangan train<->train lintas-label d<=6 (semua)
MSE kecil = duplikat sejati; MSE besar = pHash false positive.
Output: stage2b_verification.md + stage2b_crosslabel_full.csv
"""
import os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def _find_base() -> Path:
    cands = []
    if os.environ.get("DATASCAPE_BASE"):
        cands.append(Path(os.environ["DATASCAPE_BASE"]))
    cands.append(Path(r"C:/Users/PONGO AMD/Documents/codingstuff/KAGGLE_COMPETITIONS/datascape"))
    for c in cands:
        if (c / "Train_Set").is_dir():
            return c
    raise RuntimeError("base not found")



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
PHASH_THRESHOLD = 6
S = 128  # ukuran verifikasi

full = pd.read_csv(ART / "phash_full.csv", dtype={"group_id": str})
# rebuild path
def to_path(r):
    if r.split == "train":
        return str(TRAIN / r.label / r.filename)
    return str(TEST / r.filename)
full["path"] = full.apply(to_path, axis=1)

h = full.phash_hex.apply(lambda x: int(x, 16)).values.astype(np.uint64)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def pairs_with_dist(thresh, block=256):
    n = len(h)
    out = []
    for s in range(0, n, block):
        blk = h[s:s+block]
        xor = blk[:, None] ^ h[None, :]
        bts = xor.view(np.uint8).reshape(len(blk), n, 8)
        dist = POP[bts].sum(axis=2, dtype=np.uint8)
        ii, jj = np.nonzero(dist <= thresh)
        for a, c in zip(ii.tolist(), jj.tolist()):
            g = s + a
            if g < c:
                out.append((g, c, int(dist[a, c])))
    return out

t0 = time.time()
pairs = pairs_with_dist(PHASH_THRESHOLD)
print(f"pairs d<={PHASH_THRESHOLD}: {len(pairs)} [{time.time()-t0:.1f}s]")

tr_te = [(i,j,d) for i,j,d in pairs if full.split.iloc[i] != full.split.iloc[j]]
cl   = [(i,j,d) for i,j,d in pairs if full.split.iloc[i]=="train" and full.split.iloc[j]=="train"
        and full.label.iloc[i] != full.label.iloc[j]]
print(f"train<->test: {len(tr_te)} | cross-label: {len(cl)}")

def gray_mse(p):
    try:
        a = np.asarray(Image.open(p).convert("L").resize((S,S), Image.BILINEAR), dtype=np.float32)/255.
        return a
    except Exception:
        return None

def pair_mse(i, j):
    a, b = gray_mse(full.path.iloc[i]), gray_mse(full.path.iloc[j])
    if a is None or b is None: return None
    return float(((a-b)**2).mean())

def mse_stats(pair_list, label):
    with ThreadPoolExecutor(max_workers=16) as ex:
        mses = list(ex.map(lambda p: pair_mse(*p[:2]), pair_list))
    mses = [m for m in mses if m is not None]
    arr = np.array(mses)
    print(f"\n=== {label}: n={len(arr)} ===")
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q}: MSE={np.percentile(arr, q):.5f}")
    # heuristik: MSE < 0.005 ~ identik; < 0.02 sangat mirip; > 0.05 beda konten
    print(f"  identik (MSE<0.005): {(arr<0.005).sum()} | mirip (<0.02): {((arr>=0.005)&(arr<0.02)).sum()} | beda (>0.05): {(arr>0.05).sum()}")
    return arr

rng = np.random.default_rng(42)
sample_tr_te = [tr_te[i] for i in rng.choice(len(tr_te), size=min(500, len(tr_te)), replace=False)]
mse_te = mse_stats(sample_tr_te, "train<->test (sample 500)")
mse_cl = mse_stats(cl, "cross-label (semua)")

# tabel lengkap cross-label untuk inspeksi
rows = []
for (i, j, d) in cl:
    m = pair_mse(i, j)
    rows.append({"f_i": full.filename.iloc[i], "l_i": full.label.iloc[i],
                 "f_j": full.filename.iloc[j], "l_j": full.label.iloc[j],
                 "phash_d": d, "mse128": round(m, 5) if m else None})
cdf = pd.DataFrame(rows).sort_values("mse128")
cdf.to_csv(ART / "stage2b_crosslabel_full.csv", index=False)
n_true = (cdf.mse128 < 0.02).sum()
print(f"\ncross-label pairs with MSE<0.02 (kemungkinan label noise sejati): {n_true}/{len(cdf)}")
print(cdf.head(15).to_string())

rep = f"""# Stage 2b — Verifikasi numerik pasangan pHash
- train<->test d<={PHASH_THRESHOLD}: {len(tr_te)} pasangan; sample 500 diverifikasi pixel-MSE (128px grayscale)
- cross-label train<->train d<={PHASH_THRESHOLD}: {len(cl)} pasangan, SEMUA diverifikasi

## Hasil MSE train<->test (sample 500)
- p50={np.percentile(mse_te,50):.5f} p90={np.percentile(mse_te,90):.5f}
- identik (<0.005): {(mse_te<0.005).sum()} | mirip (<0.02): {((mse_te>=0.005)&(mse_te<0.02)).sum()} | beda (>0.05): {(mse_te>0.05).sum()}

## Hasil MSE cross-label (semua {len(cl)})
- p50={np.percentile(mse_cl,50):.5f} p90={np.percentile(mse_cl,90):.5f}
- identik (<0.005): {(mse_cl<0.005).sum()} | mirip (<0.02): {((mse_cl>=0.005)&(mse_cl<0.02)).sum()} | beda (>0.05): {(mse_cl>0.05).sum()}
- pasangan kemungkinan label-noise sejati (MSE<0.02): {n_true}

Kesimpulan: lihat distribusi di atas. Jika mayoritas train<->test pairs MSE kecil -> near-duplicate lintas split NYATA -> test set mengandung salinan gambar train (penting untuk interpretasi LB, TIDAK untuk curang — model tetap harus generalisasi).
Cross-label pairs dengan MSE kecil = kandidat label noise -> didaftar di stage2b_crosslabel_full.csv.
"""
(ART / "stage2b_verification.md").write_text(rep, encoding="utf-8")
print(f"\nreport: {ART/'stage2b_verification.md'} | runtime {time.time()-t0:.1f}s")

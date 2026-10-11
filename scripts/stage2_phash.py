"""DATASCAPE 2026 — Stage 2: pHash near-duplicate grouping + audit leakage.
Role (models_to_use.txt): GLM-5.3 + DeepSeek V4 Pro -> pHash grouping (phash_groups.csv)
Workflow (master_prompt_cnn_computer_vision.md): Tahap 1.2 + A.4 + input Tahap 3 (StratifiedGroupKFold).

Output (di <BASE>/artifacts/):
  phash_full.csv          hash pHash tiap gambar (hex) + group_id
  phash_groups.csv        grup near-duplicate: root, anggota (;-joined) + metadata
  image_groups.csv        siap dipakai Stage 3: filename, split, label, group_id
  stage2_dup_montage.jpg      sampel pasangan d<=THRESHOLD (verifikasi mata)
  stage2_borderline_montage.jpg  sampel pasangan d=THRESHOLD+1..10 (perlu inspeksi)
  stage2_report.md        ringkasan angka

Portable: auto-detect path lokal (Windows) atau Kaggle (/kaggle/input).
Pakai di Kaggle notebook: upload data (Train_Set/ + Test_Set/) sebagai dataset,
lalu jalankan cell: %run stage2_phash.py
"""
import os, sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ----------------------------- konfigurasi -----------------------------
def _find_base() -> Path:
    """Auto-detect root dataset (lokal Windows atau /kaggle/input)."""
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
TRAIN, TEST = BASE / "Train_Set", BASE / "Test_Set"
ART = _writable_art(BASE)

CLASSES = ["Aman", "Rusak", "Lainnya"]
SEED = 42
PHASH_THRESHOLD = 6        # Hamming <= 6 -> satu grup (workflow.md 1.2; 0-5 pasti dup, 6-10 mirip)
BORDERLINE_MAX = 10        # batas atas pasangan "borderline" untuk inspeksi mata
N_WORKERS = 16

try:
    import imagehash
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=True)
    import imagehash

# ----------------------------- util -----------------------------
def collect_files() -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        d = TRAIN / cls
        for f in sorted(os.listdir(d)):
            rows.append({"split": "train", "filename": f, "label": cls, "path": str(d / f)})
    for f in sorted(os.listdir(TEST)):
        rows.append({"split": "test", "filename": f, "label": "", "path": str(TEST / f)})
    return pd.DataFrame(rows)

def compute_phash(path):
    """pHash 64-bit -> (hex, uint64). None jika gagal decode."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            h = imagehash.phash(im, hash_size=8)
        b = np.packbits(h.hash.astype(bool)).tobytes()
        return str(h), int.from_bytes(b, "big")
    except Exception:
        return None, None

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def pairs_with_dist(h_u64, thresh, block=256):
    """Semua pasangan (i, j, d) dengan i<j dan Hamming distance d <= thresh.
    numpy chunked XOR + popcount lookup (jauh lebih cepat dari O(n^2) python)."""
    h = np.asarray(h_u64, dtype=np.uint64)
    n = len(h)
    out = []
    for s in range(0, n, block):
        blk = h[s:s + block]
        xor = blk[:, None] ^ h[None, :]
        bts = xor.view(np.uint8).reshape(len(blk), n, 8)
        dist = _POP[bts].sum(axis=2, dtype=np.uint8)
        ii, jj = np.nonzero(dist <= thresh)
        for a, c in zip(ii.tolist(), jj.tolist()):
            g = s + a
            if g < c:
                out.append((g, c, int(dist[a, c])))
    return out

class DSU:
    """Union-find; root = index terkecil -> deterministik."""
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if ra < rb: self.p[rb] = ra
        else: self.p[ra] = rb

def _thumb(path, size=224):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return im

def montage(pair_list, df, out_path, size=224):
    """Grid pasangan gambar side-by-side + header (jarak, split/label, nama file)."""
    if not pair_list:
        print(f"  (montage dilewati: tidak ada pasangan -> {out_path.name})")
        return
    H_row = size + 30
    W = 2 * (size + 12)
    canvas = Image.new("RGB", (W, H_row * len(pair_list)), "white")
    dr = ImageDraw.Draw(canvas)
    for r, (i, j, d) in enumerate(pair_list):
        y = r * H_row
        li = df.label.iloc[i] or "test"
        lj = df.label.iloc[j] or "test"
        head = f"d={d} | {df.split.iloc[i]}/{li} vs {df.split.iloc[j]}/{lj} | {df.filename.iloc[i]} ~ {df.filename.iloc[j]}"
        dr.text((6, y + 8), head[:110], fill="black")
        try:
            a, b = _thumb(df.path.iloc[i], size), _thumb(df.path.iloc[j], size)
            canvas.paste(a, (6, y + 28))
            canvas.paste(b, (size + 18, y + 28))
        except Exception as e:
            dr.text((6, y + 28), f"thumb error: {e}", fill="red")
    canvas.save(out_path, quality=88)
    print(f"  montage tersimpan: {out_path} ({len(pair_list)} pasangan)")

# ----------------------------- main -----------------------------
def main():
    t0 = time.time()
    df = collect_files()
    n_tr = int((df.split == "train").sum()); n_te = int((df.split == "test").sum())
    print(f"files: train={n_tr} test={n_te} total={len(df)}")

    # 1) hitung pHash semua gambar (threaded)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        res = list(ex.map(compute_phash, df["path"]))
    df["phash_hex"] = [r[0] for r in res]
    df["hash_u64"] = [r[1] for r in res]
    n_fail = int(df.hash_u64.isna().sum())
    print(f"pHash computed: {len(df) - n_fail}/{len(df)} | failures={n_fail} [{time.time()-t0:.1f}s]")
    assert n_fail == 0, "Ada gambar gagal di-pHash — investigasi dulu."

    # 2) pasangan dekat (d <= BORDERLINE_MAX) + distribusi jarak
    pairs = pairs_with_dist(df.hash_u64.values, BORDERLINE_MAX)
    print(f"pasangan d<={BORDERLINE_MAX}: {len(pairs)} [{time.time()-t0:.1f}s]")
    print("\n=== DISTRIBUSI JARAK HAMMING (pasangan, 64-bit pHash) ===")
    for t in [0, 2, 4, 6, 8, 10]:
        cnt = sum(1 for _, _, d in pairs if d <= t)
        print(f"  d<={t:2d}: {cnt:6d} pasangan")

    # 3) union-find -> grup pada PHASH_THRESHOLD
    dsu = DSU(len(df))
    for i, j, d in pairs:
        if d <= PHASH_THRESHOLD:
            dsu.union(i, j)
    root_idx = [dsu.find(i) for i in range(len(df))]
    root2name = {i: df.filename.iloc[i] for i in sorted(set(root_idx))}
    df["group_id"] = [root2name[r] for r in root_idx]

    # 4) tabel grup
    g = df.groupby("group_id", sort=True)
    gtab = pd.DataFrame({
        "root": g.size().index,
        "n_members": g.size().values,
        "anggota": g["filename"].apply(lambda s: ";".join(s)).values,
        "n_train": g["split"].apply(lambda s: int((s == "train").sum())).values,
        "n_test": g["split"].apply(lambda s: int((s == "test").sum())).values,
        "labels": g["label"].apply(lambda s: ";".join(sorted(set(x for x in s if x)))).values,
    })
    gtab["cross_split"] = (gtab.n_train > 0) & (gtab.n_test > 0)
    gtab["cross_label"] = gtab["labels"].str.contains(";")
    multi = gtab[gtab.n_members > 1].copy()

    # simpan artefak
    df[["filename", "split", "label", "phash_hex", "group_id"]].to_csv(ART / "phash_full.csv", index=False)
    multi[["root", "anggota", "n_members", "n_train", "n_test", "labels", "cross_split", "cross_label"]] \
        .sort_values("n_members", ascending=False).to_csv(ART / "phash_groups.csv", index=False)
    df[["filename", "split", "label", "group_id"]].to_csv(ART / "image_groups.csv", index=False)

    # 5) statistik leakage
    n_grp_total = len(gtab)
    n_grp_multi = len(multi)
    n_img_in_multi = int(multi.n_members.sum())
    n_cross_split = int(multi.cross_split.sum())
    n_cross_label = int(multi.cross_label.sum())
    cs_exact = [(i, j, d) for i, j, d in pairs
                if d == 0 and df.split.iloc[i] != df.split.iloc[j]]
    tr_te_pairs = [(i, j, d) for i, j, d in pairs
                   if d <= PHASH_THRESHOLD and df.split.iloc[i] != df.split.iloc[j]]
    cl_pairs = [(i, j, d) for i, j, d in pairs
                if d <= PHASH_THRESHOLD and df.split.iloc[i] == "train" and df.split.iloc[j] == "train"
                and df.label.iloc[i] != df.label.iloc[j]]

    print("\n=== GRUP pHash (threshold d<=%d) ===" % PHASH_THRESHOLD)
    print(f"  total grup: {n_grp_total} | multi-anggota: {n_grp_multi} | gambar dalam grup multi: {n_img_in_multi}")
    if n_grp_multi:
        print(f"  distribusi ukuran grup (top): {multi.n_members.value_counts().sort_index().to_dict()}")
    print("\n=== LEAKAGE ===")
    print(f"  grup lintas-split (train<->test): {n_cross_split}")
    print(f"  pasangan train<->test (d<=thr): {len(tr_te_pairs)} | exact d=0: {len(cs_exact)}")
    print(f"  grup lintas-label (label noise?): {n_cross_label} | pasangan: {len(cl_pairs)}")

    # 6) montase verifikasi mata
    print("\n=== MONTASE VERIFIKASI ===")
    sel_cs = tr_te_pairs[:6]
    sel_cl = cl_pairs[:6]
    sel_same = [(i, j, d) for i, j, d in pairs
                if d <= PHASH_THRESHOLD and (i, j, d) not in sel_cs and (i, j, d) not in sel_cl
                and df.split.iloc[i] == df.split.iloc[j]][:6]
    montage(sel_cs, df, ART / "stage2_dup_montage.jpg")
    montage(sel_cl, df, ART / "stage2_dup_crosslabel_montage.jpg")
    montage(sel_same, df, ART / "stage2_dup_same_montage.jpg")
    borderline = [(i, j, d) for i, j, d in pairs
                  if PHASH_THRESHOLD < d <= BORDERLINE_MAX][:12]
    montage(borderline, df, ART / "stage2_borderline_montage.jpg")

    # 7) laporan
    rep = f"""# DATASCAPE 2026 — Stage 2 Report: pHash Grouping
## Parameter
- pHash 64-bit (hash_size=8), threshold grup: Hamming <= {PHASH_THRESHOLD}
- Pasangan dihitung chunked numpy XOR+popcount, semua {len(df)} gambar (train {n_tr} + test {n_te})

## Distribusi jarak pasangan
""" + "\n".join(f"- d<={t}: {sum(1 for _,_,d in pairs if d <= t)}" for t in [0,2,4,6,8,10]) + f"""

## Grup
- total grup: {n_grp_total} | multi-anggota: {n_grp_multi} | gambar terlibat: {n_img_in_multi}
- distribusi ukuran grup multi: {multi.n_members.value_counts().sort_index().to_dict() if n_grp_multi else '-'}

## Leakage
- grup lintas-split (train<->test): {n_cross_split}
- pasangan train<->test (d<=thr): {len(tr_te_pairs)} | exact d=0: {len(cs_exact)}
- grup lintas-label (label noise?): {n_cross_label} | pasangan: {len(cl_pairs)}

## Artefak
- phash_full.csv, phash_groups.csv, image_groups.csv (input Stage 3: StratifiedGroupKFold by group_id)
- montase: stage2_dup_montage.jpg / stage2_dup_crosslabel_montage.jpg / stage2_dup_same_montage.jpg / stage2_borderline_montage.jpg

Runtime: {time.time()-t0:.1f}s
"""
    (ART / "stage2_report.md").write_text(rep, encoding="utf-8")
    print(f"\nreport tersimpan: {ART / 'stage2_report.md'}")
    print(f"TOTAL RUNTIME: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

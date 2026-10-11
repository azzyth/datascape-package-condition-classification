"""DATASCAPE 2026 — Stage 1: Load + audit integritas gambar + fast baseline.
DeepSeek V4 Pro. Deliverables: audit_report.csv, corrupt list, mode/size dist,
fast baseline (majority + mean-pixel LogisticRegression) with StratifiedKFold macro-F1.
"""
import os, sys, time, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

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
TRAIN = BASE / "Train_Set"
TEST = BASE / "Test_Set"
ART = _writable_art(BASE)

CLASSES = ["Aman", "Rusak", "Lainnya"]  # sorted; Aman=0, Rusak=1, Lainnya=2

# ---------- collect file lists ----------
train_files = []
for cls in CLASSES:
    d = TRAIN / cls
    for f in os.listdir(d):
        train_files.append((str(f), cls, str(d / f)))
test_files = [(f, "test", str(TEST / f)) for f in os.listdir(TEST)]
print(f"train={len(train_files)}  test={len(test_files)}  total={len(train_files)+len(test_files)}")

from PIL import Image

def check_image(path):
    """Fast audit: verify() (integrity) + header-only size/mode read. No full pixel decode."""
    try:
        with Image.open(path) as im:
            im.verify()          # integrity check, fast
        with Image.open(path) as im2:
            color_mode = im2.mode
            size = im2.size
        fs = os.path.getsize(path)
        return {"path": path, "status": "ok", "color_mode": color_mode,
                "w": size[0], "h": size[1], "bytes": fs}
    except Exception as e:
        return {"path": path, "status": "corrupt", "color_mode": None,
                "w": None, "h": None, "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
                "err": str(e)[:120]}

def run_parallel(items, n_workers=16):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as p:
        results = list(p.map(check_image, [it[2] for it in items]))
    print(f"  audited {len(results)} imgs in {time.time()-t0:.1f}s")
    return results

# ---------- audit ----------
print("\n=== AUDIT TRAIN ===")
tr = run_parallel(train_files)
print("=== AUDIT TEST ===")
te = run_parallel(test_files)

# merge metadata back
tr_map = {r["path"]: r for r in tr}
te_map = {r["path"]: r for r in te}

rows = []
for fname, cls, path in train_files:
    r = tr_map[path]
    rows.append({"split": "train", "filename": fname, "label": cls,
                 "status": r["status"], "color_mode": r["color_mode"], "w": r["w"], "h": r["h"],
                 "bytes": r["bytes"], "err": r.get("err", "")})
for fname, _, path in test_files:
    r = te_map[path]
    rows.append({"split": "test", "filename": fname, "label": "",
                 "status": r["status"], "color_mode": r["color_mode"], "w": r["w"], "h": r["h"],
                 "bytes": r["bytes"], "err": r.get("err", "")})
df = pd.DataFrame(rows)
df.to_csv(ART / "audit_report.csv", index=False)

# ---------- summary ----------
corrupt = df[df.status == "corrupt"]
print("\n=== SUMMARY ===")
print("total rows:", len(df))
print("corrupt:", len(corrupt))
if len(corrupt):
    print(corrupt[["split", "filename", "label", "err"]].head(30).to_string())

print("\n=== MODE DIST (train) ===")
print(df[df.split == "train"].color_mode.value_counts().to_string())
print("=== MODE DIST (test) ===")
print(df[df.split == "test"].color_mode.value_counts().to_string())

ok = df[df.status == "ok"].copy()
ok["aspect"] = ok.w / ok.h
print("\n=== SIZE (train, ok only) ===")
print(ok[ok.split == "train"][["w", "h", "aspect", "bytes"]].describe().round(1).to_string())
print("=== SIZE (test, ok only) ===")
print(ok[ok.split == "test"][["w", "h", "aspect", "bytes"]].describe().round(1).to_string())

# file name stem overlap train vs test
tr_stems = set(s.split(".")[0] for s in df[df.split == "train"].filename)
te_stems = set(s.split(".")[0] for s in df[df.split == "test"].filename)
print("\noverlap train/test stems:", len(tr_stems & te_stems))

print("\n=== DONE AUDIT ===")

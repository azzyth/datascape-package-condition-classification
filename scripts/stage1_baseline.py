"""DATASCAPE 2026 — Stage 1b: fast baseline + cheap-feature LogisticRegression.
Fitur: mean RGB, std RGB, aspect ratio, file size (dari decode 32x32).
Sekaligus = full-decode audit (decode penuh, bukan cuma verify()).
Deliverables: cheap_features.csv, baseline macro-F1, baseline submission.csv
"""
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
CLASSES = ["Aman", "Rusak", "Lainnya"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

train_files = []
for cls in CLASSES:
    for f in os.listdir(TRAIN / cls):
        train_files.append((f, cls, str(TRAIN / cls / f)))
test_files = [(f, str(TEST / f)) for f in os.listdir(TEST)]

def feats(path):
    """Decode penuh + resize 32x32, return cheap features. None jika decode gagal."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            im = im.resize((32, 32), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32) / 255.0
        mean = a.mean(axis=(0, 1))
        std = a.std(axis=(0, 1))
        return (mean[0], mean[1], mean[2], std[0], std[1], std[2], w, h, w / h, os.path.getsize(path))
    except Exception:
        return None

def extract(files):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as p:
        res = list(p.map(lambda x: feats(x[2]) if len(x) == 3 else feats(x[1]), files))
    print(f"  extracted {len(files)} in {time.time()-t0:.1f}s")
    return res

print("=== extract train ===")
tr_feats = extract(train_files)
print("=== extract test ===")
te_feats = extract(test_files)

COLS = ["r", "g", "b", "sr", "sg", "sb", "w", "h", "aspect", "bytes"]
tr_rows, te_rows = [], []
n_fail = 0
for (fname, cls, _), f in zip(train_files, tr_feats):
    if f is None:
        n_fail += 1
        continue
    tr_rows.append({"filename": fname, "label": cls, **dict(zip(COLS, f))})
for (fname, _), f in zip(test_files, te_feats):
    if f is None:
        n_fail += 1
        continue
    te_rows.append({"filename": fname, **dict(zip(COLS, f))})

trdf = pd.DataFrame(tr_rows)
tedf = pd.DataFrame(te_rows)
trdf.to_csv(ART / "cheap_features_train.csv", index=False)
tedf.to_csv(ART / "cheap_features_test.csv", index=False)
print(f"\nfull-decode FAILURES: {n_fail}")
print("train feats:", len(trdf), " test feats:", len(tedf))

# ---------- baseline ----------
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y = trdf["label"].map(CLS2IDX).values
X = trdf[COLS].values.astype(np.float32)

# 1) majority class baseline
maj = np.bincount(y).argmax()
maj_pred = np.full_like(y, maj)
maj_macro = f1_score(y, maj_pred, average="macro")
print(f"\n=== BASELINE ===")
print(f"class dist: {dict(zip(CLASSES, np.bincount(y)))}")
print(f"majority ('{CLASSES[maj]}') macro-F1 = {maj_macro:.4f}")

# 2) prior/random baseline (predict class distribution, argmax = majority anyway; use prior-weighted)
# 3) LogisticRegression cheap features
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(y), 3))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X[tr], y[tr])
    oof[va] = clf.predict_proba(X[va])
oof_pred = oof.argmax(1)
lr_macro = f1_score(y, oof_pred, average="macro")
lr_acc = accuracy_score(y, oof_pred)
per_class = f1_score(y, oof_pred, average=None)
print(f"LogReg (mean-pixel) 5-fold OOF macro-F1 = {lr_macro:.4f}  acc={lr_acc:.4f}")
print(f"per-class F1: {dict(zip(CLASSES, per_class.round(4)))}")

# confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, oof_pred)
print("confusion (rows=actual):\n", pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

# ---------- submission: use LogReg (better than majority) ----------
if len(tedf) == 3057:
    clf_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf_full.fit(X, y)
    Xte = tedf[COLS].values.astype(np.float32)
    te_prob = clf_full.predict_proba(Xte)
    te_pred = te_prob.argmax(1)
    sub = pd.DataFrame({"filename": tedf["filename"], "label": [CLASSES[i] for i in te_pred]})
    # reorder to match sample_submission
    ss = pd.read_csv(BASE / "sample_submission.csv")
    sub = ss[["filename"]].merge(sub, on="filename", how="left")
    sub["label"] = sub["label"].fillna(CLASSES[maj])
    sub.to_csv(ART / "submission_baseline_logreg.csv", index=False)
    print(f"\nsubmission saved: {len(sub)} rows, dist: {sub.label.value_counts().to_dict()}")
    # sanity: assert format vs sample_submission
    assert list(sub.columns) == list(ss.columns), "col mismatch"
    assert len(sub) == len(ss), "row mismatch"
    assert (sub.filename.values == ss.filename.values).all(), "order mismatch"
    assert sub.label.isin(CLASSES).all(), "bad label"
    print("format assert PASSED vs sample_submission")
else:
    print(f"WARNING: test feats {len(tedf)} != 3057, check decode failures")

print("\n=== DONE BASELINE ===")

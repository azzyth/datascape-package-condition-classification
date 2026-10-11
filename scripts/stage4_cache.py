"""DATASCAPE 2026 — Stage 4: Cache Resize PAD-768 (Jendela 2 / Data Engine).

Workflow: master_prompt_cnn_computer_vision.md Tahap 4.4 (caching decode).
Tujuan: decode + pad_to_square (cap 768) SEKALI -> JPEG q95. Training berikutnya
membaca cache (4x lebih cepat, decode JPEG berulang = bottleneck terbesar).

Output:
  <BASE>/cache_768/Train_Set/{Aman,Rusak,Lainnya}/{stem}.jpg
  <BASE>/cache_768/Test_Set/{stem}.jpg
  <BASE>/artifacts/cache_manifest.csv      (filename, orig w/h, cache path/bytes)
  <BASE>/artifacts/stage4_cache_report.md

Pemakaian:
  python stage4_cache.py                 # build penuh (resumable — skip yang sudah ada)
  python stage4_cache.py --limit 50       # uji cepat
  python stage4_cache.py --verify-only    # hanya verifikasi cache yang ada
Lalu training: kaggle_all_in_one.py --stage train --cache <BASE>/cache_768

Catatan desain:
- pad_to_square IDENTIK dengan stage5 (cap 768, BILINEAR, pad hitam) -> transform
  training tidak berubah sama sekali (PadToSquare jadi no-op pada gambar persegi).
- JPEG q95: hampir lossless; training sudah punya RandomJPEGCompression(30-95)
  sehingga robust terhadap artefak kompresi. Eval val & test sama-sama pakai cache
  -> konsisten (test-time parity terjaga).
- Resumable: file cache yang sudah ada (size>0) di-skip.
"""
import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

CLASSES = ["Aman", "Rusak", "Lainnya"]
PAD_CAP = 768
JPEG_QUALITY = 95


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
    art = base / "artifacts"
    try:
        art.mkdir(parents=True, exist_ok=True)
        probe = art / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return art
    except OSError:
        out = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/content/working")
        art2 = out / "artifacts"
        art2.mkdir(parents=True, exist_ok=True)
        print(f"[path] {base} read-only -> artefak ditulis ke {art2}")
        return art2


def pad_to_square(img, cap=PAD_CAP):
    """IDENTIK stage5: convert RGB, downscale >cap, pad hitam ke persegi."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="uji cepat: hanya N file pertama per split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    base = _find_base()
    art = _writable_art(base)
    cache = base / "cache_768"
    t0 = time.time()

    # kumpulkan pekerjaan: (src_path, dst_path)
    jobs = []
    for cls in CLASSES:
        d = base / "Train_Set" / cls
        for f in sorted(os.listdir(d)):
            jobs.append((d / f, cache / "Train_Set" / cls / (Path(f).stem + ".jpg")))
    for f in sorted(os.listdir(base / "Test_Set")):
        jobs.append((base / "Test_Set" / f, cache / "Test_Set" / (Path(f).stem + ".jpg")))
    if args.limit:
        n_tr = min(args.limit, sum(1 for s, _ in jobs if "Train_Set" in str(s)))
        jobs = [j for j in jobs if "Train_Set" in str(j[0])][:n_tr] + \
               [j for j in jobs if "Test_Set" in str(j[0])][:args.limit]
    print(f"[cache] base={base} | {len(jobs)} file target | cache={cache}")

    if not args.verify_only:
        def process(job):
            src, dst = job
            if dst.is_file() and dst.stat().st_size > 0:
                return ("skip", src, dst, 0)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = pad_to_square(Image.open(src))
            img.save(dst, format="JPEG", quality=JPEG_QUALITY)
            return ("ok", src, dst, dst.stat().st_size)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(process, jobs))
        n_ok = sum(1 for r in results if r[0] == "ok")
        n_skip = sum(1 for r in results if r[0] == "skip")
        print(f"[cache] dibuat {n_ok} | skip (sudah ada) {n_skip} [{time.time()-t0:.1f}s]")

        pd.DataFrame([{
            "src": str(r[1]), "cache": str(r[2]), "status": r[0], "bytes": r[3]
        } for r in results]).to_csv(art / "cache_manifest.csv", index=False)

    # verifikasi: sampel acak — persegi, RGB, bisa dibuka
    rng = np.random.default_rng(42)
    all_cached = [dst for _, dst in jobs if dst.is_file()]
    assert all_cached, "cache kosong!"
    sample = rng.choice(len(all_cached), size=min(30, len(all_cached)), replace=False)
    errs = []
    for i in sample:
        p = all_cached[int(i)]
        with Image.open(p) as im:
            if im.size[0] != im.size[1]:
                errs.append(f"{p.name}: {im.size} bukan persegi")
            if im.mode != "RGB":
                errs.append(f"{p.name}: mode {im.mode}")
    n_expected = len(jobs)
    n_missing = n_expected - len(all_cached)
    print(f"[verify] {len(all_cached)}/{n_expected} cached | hilang: {n_missing} | sampel cek: {len(sample)} | error: {len(errs)}")
    for e in errs[:5]:
        print("  -", e)
    assert not errs, "cache corrupt!"
    assert n_missing == 0, f"{n_missing} file belum ter-cache (jalankan ulang tanpa --limit)"

    # laporan
    total_mb = sum(d.stat().st_size for _, d in jobs if d.is_file()) / 1e6
    rep = art / "stage4_cache_report.md"
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"# Stage 4 — Cache PAD-768 Report\n\n"
                f"- File cached: {len(all_cached)}/{n_expected}\n"
                f"- Ukuran total: {total_mb:.1f} MB | JPEG q{JPEG_QUALITY}, pad cap {PAD_CAP}\n"
                f"- Verifikasi sampel {len(sample)}: persegi+RGB OK, 0 error\n"
                f"- Waktu build: {time.time()-t0:.1f}s\n"
                f"- Pemakaian: `kaggle_all_in_one.py --stage train --cache {cache}`\n")
    print(f"[report] {rep}")
    print(f"STAGE 4 CACHE OK [{time.time()-t0:.1f}s] — training selanjutnya 4x lebih cepat")


if __name__ == "__main__":
    main()

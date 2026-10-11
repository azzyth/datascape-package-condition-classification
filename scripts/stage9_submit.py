#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 9 — SUBMISSION & FORMAT VALIDATION (J4)  —  DATASCAPE 2026
==================================================================
Assert vs sample_submission + sanity distribusi + (opsional) submit via Kaggle API.
[master_prompt TAHAP 9 + GATE-4]  Submit HANYA via J4.

MODE:
  1. validate  : assert CSV submission vs sample_submission (tanpa submit)
  2. build     : test_probs.npz -> argmax -> CSV -> assert (tanpa submit)
  3. submit    : build + assert + upload via Kaggle API + log submission_log.csv

ATURAN HARD (GATE-4):
  - shape/kolom/URUTAN kolom/urutan baris/dtype/NaN/inf/label valid/ID cocok persis
  - sanity distribusi: tidak boleh 100% satu kelas; proporsi mirip train
  - log setiap submit ke artifacts/submission_log.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CLASSES = ["Aman", "Rusak", "Lainnya"]  # Aman=0, Rusak=1, Lainnya=2
# Proporsi train (untuk sanity): Aman 72.0% / Rusak 24.5% / Lainnya 3.5%
TRAIN_PROP = {"Aman": 0.720, "Rusak": 0.245, "Lainnya": 0.035}
SLUG = "datascape-2026-data-mining-competition"


def resolve_base():
    env = os.environ.get("DATASCAPE_BASE")
    if env:
        return env
    if os.path.isdir("/kaggle/input"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return os.environ.get("DATASCAPE_OUT", "/content/working")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def assert_submission(sub_path, sample_path):
    """Return (ok, list_of_errors)."""
    errs = []
    sub = pd.read_csv(sub_path, keep_default_na=False)
    samp = pd.read_csv(sample_path, keep_default_na=False)

    if list(sub.columns) != list(samp.columns):
        errs.append(f"kolom {list(sub.columns)} != {list(samp.columns)}")
    if len(sub) != len(samp):
        errs.append(f"baris {len(sub)} != {len(samp)}")
    if not np.array_equal(sub["filename"].to_numpy(), samp["filename"].to_numpy()):
        errs.append("urutan/isi 'filename' TIDAK cocok sample_submission")
    if sub["label"].isna().any():
        errs.append("ada label kosong/NaN")
    if sub["filename"].isna().any():
        errs.append("ada filename kosong/NaN")
    bad = set(sub["label"].unique()) - set(CLASSES)
    if bad:
        errs.append(f"label tak dikenal: {bad}")

    # sanity distribusi
    vc = sub["label"].value_counts(normalize=True)
    if len(vc) == 1:
        errs.append("RED FLAG: prediksi 100% satu kelas")
    else:
        for c, exp in TRAIN_PROP.items():
            got = vc.get(c, 0.0)
            if abs(got - exp) > 0.25:
                errs.append(f"proporsi {c} = {got:.3f} jauh dari train {exp:.3f}")

    return (len(errs) == 0), errs


def build_submission(test_probs_path, out_path):
    z = np.load(test_probs_path, allow_pickle=False)
    ids = z["ids"].astype(str)
    probs = z["probs"]
    pred = probs.argmax(1)
    labels = [CLASSES[i] for i in pred]
    df = pd.DataFrame({"filename": ids, "label": labels})
    df.to_csv(out_path, index=False)
    return df


def refresh_oauth():
    """Refresh token OAuth kaggle jika 401 (lihat ORCHESTRATION.md)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception:
        pass
    # fallback: kagglesdk refresh_access_token
    try:
        from kaggle_secrets import UserSecretsClient  # notebook Kaggle
        return None
    except Exception:
        pass
    return None


def submit_via_cli(sub_path, message):
    """Upload via `kaggle competitions submit` CLI (butuh OAuth credentials.json)."""
    cmd = ["kaggle", "competitions", "submit", "-c", SLUG,
           "-f", sub_path, "-m", message]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


# Skema submission_log.csv = MILIK J1 (lihat PROMPT_J1.md). J4 HANYA append
# baris dengan kolom yang SAMA, tidak boleh mengubah header.
SUBMISSION_LOG_COLS = ["submission_id", "filename", "date", "stage", "model",
                       "description", "cv_score", "oof_score", "lb_public",
                       "lb_private", "note"]


def log_submission(base, candidate, basis_cv, lb="", status="", note="",
                   filename="", submission_id="", oof_score=""):
    log_path = os.path.join(base, "artifacts", "submission_log.csv")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {
        "submission_id": submission_id,   # diisi J1 dari LB (ref number)
        "filename": filename,
        "date": now,
        "stage": 9,
        "model": candidate,
        "description": note,              # pesan/deskripsi kandidat
        "cv_score": basis_cv,             # basis skor CV (mis. OOF ensemble)
        "oof_score": oof_score,
        "lb_public": lb,                  # diisi setelah LB (J1/J4 fetch)
        "lb_private": "",
        "note": status,                   # SUBMITTED / VALIDATED / FAILED(n)
    }
    exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_LOG_COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"Logged -> {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "build", "submit"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--probs", default=None, help="path *_test.npz / test_probs_blend.npz")
    ap.add_argument("--sub", default=None, help="path CSV submission untuk validate")
    ap.add_argument("--out", default=None, help="path CSV output")
    ap.add_argument("--candidate", default="ensemble", help="nama kandidat untuk log")
    ap.add_argument("--basis-cv", default="", help="basis skor CV untuk log")
    ap.add_argument("--message", default="J4 submit", help="pesan upload Kaggle")
    args = ap.parse_args()

    base = args.base or resolve_base()
    sample = os.path.join(base, "sample_submission.csv")
    if not os.path.isfile(sample):
        sys.exit(f"sample_submission.csv tidak ditemukan di {sample}")

    if args.mode == "validate":
        sub = args.sub
        if not sub:
            sys.exit("--sub wajib untuk mode validate")
    else:
        probs = args.probs
        if not probs:
            # default: hasil blend stage7
            probs = os.path.join(base, "artifacts", "preds", "ensemble",
                                 "test_probs_blend.npz")
        sub = args.out or os.path.join(base, "artifacts",
                                       f"submission_{args.candidate}.csv")
        df = build_submission(probs, sub)
        print(f"Submission dibangun: {sub} ({len(df)} baris)")

    ok, errs = assert_submission(sub, sample)
    if not ok:
        print("[ASSERT GAGAL]")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("[ASSERT PASS] format vs sample_submission OK")

    if args.mode == "submit":
        rc, out = submit_via_cli(sub, args.message)
        status = "SUBMITTED" if rc == 0 else f"FAILED({rc})"
        print(f"Kaggle submit: {status}\n{out}")
        log_submission(base, args.candidate, args.basis_cv, lb="", status=status,
                       note=args.message, filename=os.path.basename(sub))
        sys.exit(rc)
    else:
        # log validasi lokal (status VALIDATED)
        log_submission(base, args.candidate, args.basis_cv, lb="", status="VALIDATED",
                       note=f"assert pass; file={os.path.basename(sub)}",
                       filename=os.path.basename(sub))


if __name__ == "__main__":
    main()

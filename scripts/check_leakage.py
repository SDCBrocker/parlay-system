"""
check_leakage.py — Cek leakage otomatis di dataset.
Input : data/processed/matches_normalized.csv
Output: logs/check_leakage.txt
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES


def check_high_correlation(df, labels=["result_enc", "over_2_5", "btts"], threshold=0.95):
    """Cek fitur dengan korelasi > threshold ke label."""
    warnings = []
    for label in labels:
        if label not in df.columns:
            continue
        for feat in FEATURES:
            if feat not in df.columns:
                continue
            try:
                corr = df[feat].astype(float).corr(df[label].astype(float))
                if abs(corr) > threshold:
                    warnings.append(f"{feat} ↔ {label}: korelasi {corr:.4f}")
            except Exception:
                pass
    return warnings


def check_future_columns(df):
    """Cek kolom yang mengandung kata 'future' / 'next' / 'final'."""
    suspicious = []
    keywords = ["future", "next", "final", "end_", "post_", "after_"]
    for col in df.columns:
        for kw in keywords:
            if kw in col.lower():
                suspicious.append(col)
                break
    return suspicious


def check_rolling_shift(df):
    """Cek fitur rolling pakai shift (baris pertama NaN)."""
    warnings = []
    rolling_cols = [c for c in df.columns if "form_" in c or "avg_5" in c]
    for col in rolling_cols:
        if col in df.columns:
            first_val = df[col].iloc[0] if len(df) > 0 else None
            if pd.notna(first_val) and first_val != 0:
                warnings.append(f"{col}: baris pertama = {first_val} (indikasi tidak shift)")
    return warnings


def main():
    print("=== CEK LEAKAGE OTOMATIS ===")

    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        print(f"[ERROR] {p} tidak ada")
        return

    df = pd.read_csv(p, low_memory=False)
    print(f"Data: {len(df)} baris, {len(df.columns)} kolom")

    if "result" in df.columns and "result_enc" not in df.columns:
        df["result_enc"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    all_warnings = []

    # 1. High correlation
    print("\n[1] Cek korelasi tinggi (>0.95)...")
    corr_warn = check_high_correlation(df)
    if corr_warn:
        for w in corr_warn:
            print(f"  ⚠ {w}")
        all_warnings.extend(corr_warn)
    else:
        print("  ✓ Tidak ada")

    # 2. Future columns
    print("\n[2] Cek kolom 'future'/'next'/'final'...")
    fut_warn = check_future_columns(df)
    if fut_warn:
        for w in fut_warn:
            print(f"  ⚠ {w}")
        all_warnings.extend(fut_warn)
    else:
        print("  ✓ Tidak ada")

    # 3. Rolling shift
    print("\n[3] Cek rolling pakai shift...")
    shift_warn = check_rolling_shift(df)
    if shift_warn:
        for w in shift_warn[:10]:
            print(f"  ⚠ {w}")
        all_warnings.extend(shift_warn)
    else:
        print("  ✓ Tidak ada")

    # Simpan
    out = path("logs/check_leakage.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("=== CEK LEAKAGE ===\n\n")
        if all_warnings:
            for w in all_warnings:
                f.write(f"⚠ {w}\n")
        else:
            f.write("✓ Tidak ada leakage terdeteksi\n")

    print(f"\n[OK] Disimpan: {out}")
    print(f"[INFO] Total warning: {len(all_warnings)}")


if __name__ == "__main__":
    main()

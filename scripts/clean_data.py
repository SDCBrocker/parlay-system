"""
clean_data.py — Bersihkan data mentah.
Input : data/raw/history/matches_history.csv
Output: data/processed/matches_clean.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


# Kolom yang menyebabkan leakage (hasil akhir musim)
LEAKAGE_KEYWORDS = [
    "SeasonPos", "SeasonPoints", "SeasonWin", "SeasonDraw", "SeasonLoss",
    "SeasonGF", "SeasonGA", "SeasonGD",
    "final", "end_position", "promoted", "relegated",
]


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Buang kolom yang mengandung keyword leakage."""
    cols_to_drop = []
    for col in df.columns:
        for kw in LEAKAGE_KEYWORDS:
            if kw.lower() in col.lower():
                cols_to_drop.append(col)
                break
    if cols_to_drop:
        print(f"  Buang {len(cols_to_drop)} kolom leakage:")
        for c in cols_to_drop:
            print(f"    - {c}")
        df = df.drop(columns=cols_to_drop)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop kolom >50% kosong, isi sisanya."""
    # Drop kolom >50% kosong
    threshold = len(df) * 0.5
    before = len(df.columns)
    df = df.dropna(axis=1, thresh=threshold)
    after = len(df.columns)
    if before != after:
        print(f"  Drop {before - after} kolom >50% kosong")

    # Isi numerik dengan 0
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Isi kategorikal dengan "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi date ke datetime, sort, reset index."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    print("=== CLEAN DATA ===")

    in_file = path("data/raw/history/matches_history.csv")
    out_file = path("data/processed/matches_clean.csv")

    if not in_file.exists():
        print(f"❌ Input tidak ada: {in_file}")
        print("   Jalankan dulu: python scripts/fetch_history.py")
        return

    df = pd.read_csv(in_file, low_memory=False)
    print(f"Input: {len(df)} baris, {len(df.columns)} kolom")

    df = drop_leakage_columns(df)
    df = handle_missing(df)
    df = convert_date(df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"\n✓ Output: {out_file}")
    print(f"✓ {len(df)} baris, {len(df.columns)} kolom")
    print(f"✓ Rentang: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()

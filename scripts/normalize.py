"""
normalize.py — Normalisasi antar liga.
Input : data/processed/matches_features.csv
Output: data/processed/matches_normalized.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


# Koefisien kekuatan liga (relatif terhadap EPL = 1.00)
LEAGUE_COEF = {
    "EPL":          1.00,
    "LaLiga":       0.98,
    "SerieA":       0.96,
    "Bundesliga":   0.95,
    "Ligue1":       0.92,
    "Eredivisie":   0.85,
    "PrimeiraLiga": 0.84,
    "Championship": 0.80,
    "LigaMX":       0.78,
    "Brasileirao":  0.76,
}

# Kolom yang akan dinormalisasi
COLS_TO_NORMALIZE = [
    "goals_for", "goals_against",
    "home_goals", "away_goals",
    "home_shots", "away_shots",
    "home_corners", "away_corners",
]

# Kolom yang TIDAK dinormalisasi (sudah relatif)
COLS_SKIP = [
    "possession", "passes_accuracy",
    "form_5", "points_per_game",
    "is_home",
]


def add_league_coef(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom league_coef."""
    df["league_coef"] = df["league"].map(LEAGUE_COEF).fillna(1.0)
    return df


def zscore_per_league(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score per (league, season) untuk COLS_TO_NORMALIZE.
    Hasil: {col}_z
    """
    for col in COLS_TO_NORMALIZE:
        if col not in df.columns:
            continue
        z_col = f"{col}_z"
        df[z_col] = (
            df.groupby(["league", "season"])[col]
            .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        )
        df[z_col] = df[z_col].fillna(0)
    return df


def adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusted = z-score × league_coef.
    Hasil: {col}_adjusted
    """
    for col in COLS_TO_NORMALIZE:
        z_col = f"{col}_z"
        if z_col not in df.columns:
            continue
        adj_col = f"{col}_adjusted"
        df[adj_col] = df[z_col] * df["league_coef"]
    return df


def main():
    print("=== NORMALISASI ===")

    in_file = path("data/processed/matches_features.csv")
    out_file = path("data/processed/matches_normalized.csv")

    if not in_file.exists():
        print(f"❌ Input tidak ada: {in_file}")
        print("   Jalankan dulu: python scripts/features.py")
        return

    df = pd.read_csv(in_file, low_memory=False)
    print(f"Input: {len(df)} baris")

    df = add_league_coef(df)
    df = zscore_per_league(df)
    df = adjusted_features(df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"\n✓ Output: {out_file}")
    print(f"✓ {len(df)} baris, {len(df.columns)} kolom")
    print(f"✓ Liga: {sorted(df['league'].unique())}")


if __name__ == "__main__":
    main()

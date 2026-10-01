"""
schema.py — Definisi skema kolom standar untuk matches_normalized.csv.
Semua script fetch WAJIB menghasilkan kolom sesuai ini.

Versi 3: menggunakan central league registry, hindari duplikasi & inconsistency.
"""
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from league_registry import LEAGUE_REGISTRY, FD_LEAGUE_CODES

import sys
from pathlib import Path


# === Kolom wajib (target akhir) ===
REQUIRED_COLUMNS = [
    # Identitas
    "match_id",       # string unik
    "date",           # datetime
    "league",         # EPL, LaLiga, dst
    "season",         # 2023, 2024, dst
    "home_team",      # string
    "away_team",      # string

    # Hasil (label training)
    "home_goals",     # int
    "away_goals",     # int
    "result",         # H / D / A
    "over_1_5",       # 0/1
    "over_2_5",       # 0/1
    "over_3_5",       # 0/1
    "btts",           # 0/1

    # Statistik (jika tersedia)
    "home_shots", "away_shots",
    "home_sot", "away_sot",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",

    # Odds (jika tersedia)
    "odds_home", "odds_draw", "odds_away",
    "odds_over_2_5", "odds_under_2_5",

    # Metadata
    "source",         # "history" / "api"
]


# === Mapping dari football-data.co.uk CSV ===
# Ref: https://www.football-data.co.uk/notes.txt
FD_MAPPING = {
    "Date":     "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":     "home_goals",
    "FTAG":     "away_goals",
    "FTR":      "result",
    "HS":       "home_shots",
    "AS":       "away_shots",
    "HST":      "home_sot",
    "AST":      "away_sot",
    "HC":       "home_corners",
    "AC":       "away_corners",
    "HF":       "home_fouls",
    "AF":       "away_fouls",
    # Odds Bet365 (mulai musim 2002/03)
    "B365H":    "odds_home",
    "B365D":    "odds_draw",
    "B365A":    "odds_away",
    "B365>2.5": "odds_over_2_5",
    "B365<2.5": "odds_under_2_5",
}


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paksa df memiliki semua REQUIRED_COLUMNS.
    Kolom yang tidak ada → NaN.
    Kolom ekstra → dibuang.

    Versi optimasi: hindari fragmentasi DataFrame dengan
    membuat kolom kosong sekaligus via pd.concat, bukan satu-satu.
    """
    # Kumpulkan kolom yang sudah ada & yang hilang
    existing = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    # Ambil kolom yang ada
    df_existing = df[existing].copy()

    # Buat DataFrame baru untuk kolom yang hilang sekaligus
    if missing:
        df_missing = pd.DataFrame(
            {col: pd.NA for col in missing},
            index=df.index,
        )
        result = pd.concat([df_existing, df_missing], axis=1)
    else:
        result = df_existing

    # Urutkan sesuai REQUIRED_COLUMNS
    return result[REQUIRED_COLUMNS].copy()


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung kolom turunan: result, over_1_5, over_2_5, over_3_5, btts.
    Hanya jika kolom sumber tersedia.
    """
    if "home_goals" in df.columns and "away_goals" in df.columns:
        # Konversi ke numerik dulu (antisipasi string)
        hg = pd.to_numeric(df["home_goals"], errors="coerce")
        ag = pd.to_numeric(df["away_goals"], errors="coerce")

        total = hg + ag

        df["over_1_5"] = (total > 1.5).astype("Int64")
        df["over_2_5"] = (total > 2.5).astype("Int64")
        df["over_3_5"] = (total > 3.5).astype("Int64")
        df["btts"] = ((hg > 0) & (ag > 0)).astype("Int64")

        # Result kalau belum ada / kosong
        if "result" not in df.columns or df["result"].isna().all():
            df["result"] = "D"
            df.loc[hg > ag, "result"] = "H"
            df.loc[hg < ag, "result"] = "A"

    return df


if __name__ == "__main__":
    print("REQUIRED_COLUMNS:", len(REQUIRED_COLUMNS), "kolom")
    print("FD_MAPPING:", len(FD_MAPPING), "mapping")
    print("FD_LEAGUE_CODES:", FD_LEAGUE_CODES)
    print()

    # Test ensure_schema
    test_df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "FTHG": [2],
        "FTAG": [1],
        "RandomExtraColumn": ["xxx"],
    })
    test_df = test_df.rename(columns=FD_MAPPING)
    result = ensure_schema(test_df)
    print("Test ensure_schema:")
    print(f"  Input kolom : {list(test_df.columns)}")
    print(f"  Output kolom: {list(result.columns)}")
    print(f"  Jumlah      : {len(result.columns)} (harus {len(REQUIRED_COLUMNS)})")
    print()
    result = derive_labels(result)
    print("Test derive_labels:")
    print(f"  result     : {result['result'].iloc[0]}")
    print(f"  over_1_5   : {result['over_1_5'].iloc[0]}")
    print(f"  over_2_5   : {result['over_2_5'].iloc[0]}")
    print(f"  over_3_5   : {result['over_3_5'].iloc[0]}")
    print(f"  btts       : {result['btts'].iloc[0]}")

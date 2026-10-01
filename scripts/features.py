"""
features.py — Feature engineering.
Input : data/processed/matches_clean.csv
Output: data/processed/matches_features.csv
Versi 5: Tambah consistency & rest-day features ke main(), perbaiki congestion calculation.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


def create_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """goal_diff = home_goals - away_goals, dll."""
    if "home_goals" in df and "away_goals" in df:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
    if "home_xg" in df and "away_xg" in df:
        df["xg_diff"] = df["home_xg"] - df["away_xg"]
    return df


def _team_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ubah df dari format '1 baris = 1 laga' jadi '1 baris = 1 tim per laga'.
    Supaya rolling bisa group by team.
    """
    home = df[["date", "league", "season", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "league", "season", "team", "goals_for", "goals_against"]
    home["is_home"] = 1

    away = df[["date", "league", "season", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "league", "season", "team", "goals_for", "goals_against"]
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling form per tim. WAJIB shift(1)."""
    long = _team_long_df(df)

    # Poin: menang=3, seri=1, kalah=0
    long["points"] = 0
    long.loc[long["goals_for"] > long["goals_against"], "points"] = 3
    long.loc[long["goals_for"] == long["goals_against"], "points"] = 1

    # Shift(1) supaya tidak leakage
    long["points_shift"] = long.groupby("team")["points"].shift(1)
    long["gf_shift"] = long.groupby("team")["goals_for"].shift(1)
    long["ga_shift"] = long.groupby("team")["goals_against"].shift(1)

    # Rolling
    for w in [5, 10]:
        long[f"form_points_{w}"] = (
            long.groupby("team")["points_shift"]
            .rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
    long["goals_for_avg_5"] = (
        long.groupby("team")["gf_shift"].rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    long["goals_against_avg_5"] = (
        long.groupby("team")["ga_shift"].rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Ambil kolom yang perlu, merge balik ke df utama
    feat_cols = ["date", "team", "form_points_5", "form_points_10",
                 "goals_for_avg_5", "goals_against_avg_5"]
    long_feat = long[feat_cols].copy()

    # Merge untuk home
    home_feat = long_feat.rename(columns={
        "team": "home_team",
        "form_points_5": "home_form_5",
        "form_points_10": "home_form_10",
        "goals_for_avg_5": "home_goals_for_avg_5",
        "goals_against_avg_5": "home_goals_against_avg_5",
    })
    df = df.merge(home_feat, on=["date", "home_team"], how="left")

    # Merge untuk away
    away_feat = long_feat.rename(columns={
        "team": "away_team",
        "form_points_5": "away_form_5",
        "form_points_10": "away_form_10",
        "goals_for_avg_5": "away_goals_for_avg_5",
        "goals_against_avg_5": "away_goals_against_avg_5",
    })
    df = df.merge(away_feat, on=["date", "away_team"], how="left")

    return df


def create_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """Home win rate & away win rate per tim (rolling)."""
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
    df["away_win"] = (df["away_goals"] > df["home_goals"]).astype(int)

    # Rolling home win rate per tim (shift 1)
    df = df.sort_values("date").reset_index(drop=True)
    df["home_win_rate"] = (
        df.groupby("home_team")["home_win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["away_win_rate"] = (
        df.groupby("away_team")["away_win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    return df


def create_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head 5 pertemuan terakhir."""
    df = df.sort_values("date").reset_index(drop=True)

    # Buat pasangan tim (sorted) untuk group H2H
    df["pair"] = df.apply(
        lambda r: "_".join(sorted([str(r["home_team"]), str(r["away_team"])])),
        axis=1,
    )
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["home_won"] = (df["home_goals"] > df["away_goals"]).astype(int)

    df["h2h_avg_goals"] = (
        df.groupby("pair")["total_goals"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["h2h_home_wins"] = (
        df.groupby("pair")["home_won"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    return df


def create_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_home = 1 (untuk konsistensi dengan data tim)."""
    df["is_home"] = 1
    return df


def create_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur konsistensi tim (std dev, clean sheet, failed to score)."""
    home = df[["date", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "gf", "ga"]
    away = df[["date", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "gf", "ga"]

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    long["points"] = 0
    long.loc[long["gf"] > long["ga"], "points"] = 3
    long.loc[long["gf"] == long["ga"], "points"] = 1

    long["clean_sheet"] = (long["ga"] == 0).astype(int)
    long["failed_score"] = (long["gf"] == 0).astype(int)

    # Shift(1) untuk hindari leakage
    for col in ["points", "clean_sheet", "failed_score", "gf", "ga"]:
        long[f"{col}_shift"] = long.groupby("team")[col].shift(1)

    long["form_std_5"] = long.groupby("team")["points_shift"].rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
    long["clean_sheet_rate_5"] = long.groupby("team")["clean_sheet_shift"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    long["failed_score_rate_5"] = long.groupby("team")["failed_score_shift"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    feat_cols = ["date", "team", "form_std_5", "clean_sheet_rate_5", "failed_score_rate_5"]
    long_feat = long[feat_cols].copy()

    home_feat = long_feat.rename(columns={
        "team": "home_team",
        "form_std_5": "home_form_std_5",
        "clean_sheet_rate_5": "home_clean_sheet_rate_5",
        "failed_score_rate_5": "home_failed_score_rate_5",
    })
    df = df.merge(home_feat, on=["date", "home_team"], how="left")

    away_feat = long_feat.rename(columns={
        "team": "away_team",
        "form_std_5": "away_form_std_5",
        "clean_sheet_rate_5": "away_clean_sheet_rate_5",
        "failed_score_rate_5": "away_failed_score_rate_5",
    })
    df = df.merge(away_feat, on=["date", "away_team"], how="left")

    return df


def create_rest_days_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur rest days & congestion (fixed calculation)."""
    df = df.sort_values("date").reset_index(drop=True)

    # Rest days home & away
    df["home_rest_days"] = df.groupby("home_team")["date"].diff().dt.days
    df["away_rest_days"] = df.groupby("away_team")["date"].diff().dt.days

    # Congestion 7 hari terakhir (fixed: count actual matches in 7d window)
    def count_matches_in_7d(group_df, current_idx):
        """Count berapa banyak matches tim dalam 7 hari sebelum current_date."""
        current_date = group_df.iloc[current_idx]["date"]
        window_start = current_date - pd.Timedelta(days=7)
        # Count matches >= 7 days ago but < current date
        count = ((group_df["date"] > window_start) & (group_df["date"] < current_date)).sum()
        return count

    # Home congestion
    home_congestion = []
    for team in df["home_team"].unique():
        team_home_df = df[df["home_team"] == team].reset_index(drop=True)
        team_congestion = []
        for idx in range(len(team_home_df)):
            count = count_matches_in_7d(team_home_df, idx)
            team_congestion.append(count)
        home_congestion.extend(team_congestion)
    
    # Away congestion
    away_congestion = []
    for team in df["away_team"].unique():
        team_away_df = df[df["away_team"] == team].reset_index(drop=True)
        team_congestion = []
        for idx in range(len(team_away_df)):
            count = count_matches_in_7d(team_away_df, idx)
            team_congestion.append(count)
        away_congestion.extend(team_congestion)
    
    # Assign kembali ke dataframe (hati-hati dengan urutan)
    df_temp = df.copy()
    df_temp["home_congestion_7d"] = 0
    df_temp["away_congestion_7d"] = 0
    
    for team in df["home_team"].unique():
        mask = df["home_team"] == team
        team_df = df[mask].reset_index(drop=True)
        congestion_vals = []
        for idx in range(len(team_df)):
            count = count_matches_in_7d(team_df, idx)
            congestion_vals.append(count)
        df.loc[mask, "home_congestion_7d"] = congestion_vals
    
    for team in df["away_team"].unique():
        mask = df["away_team"] == team
        team_df = df[mask].reset_index(drop=True)
        congestion_vals = []
        for idx in range(len(team_df)):
            count = count_matches_in_7d(team_df, idx)
            congestion_vals.append(count)
        df.loc[mask, "away_congestion_7d"] = congestion_vals

    return df


def check_rolling_leakage(df: pd.DataFrame):
    """Warning kalau baris pertama rolling tidak NaN."""
    for col in ["home_form_5", "away_form_5"]:
        if col in df.columns:
            first_val = df[col].iloc[0]
            if pd.notna(first_val) and first_val != 0:
                print(f"  ⚠ Warning: {col} baris pertama = {first_val} (indikasi leakage?)")


def main():
    print("=== FEATURE ENGINEERING ===")

    in_file = path("data/processed/matches_clean.csv")
    out_file = path("data/processed/matches_features.csv")

    if not in_file.exists():
        print(f"❌ Input tidak ada: {in_file}")
        print("   Jalankan dulu: python scripts/clean_data.py")
        return

    df = pd.read_csv(in_file, low_memory=False)
    print(f"Input: {len(df)} baris")

    df = create_differential_features(df)
    df = create_rolling_features(df)
    df = create_home_away_features(df)
    df = create_h2h_features(df)
    df = create_contextual_features(df)
    
    # FIX #1.1: Panggil consistency & rest-day features
    print("\n[Stage 1 Fix #1.1] Adding consistency features...")
    df = create_consistency_features(df)
    print("  ✓ Consistency features added")
    
    print("[Stage 1 Fix #1.1] Adding rest-day & congestion features...")
    df = create_rest_days_features(df)
    print("  ✓ Rest-day & congestion features added")

    check_rolling_leakage(df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"\n✓ Output: {out_file}")
    print(f"✓ {len(df)} baris, {len(df.columns)} kolom")
    print(f"✓ Features: {sorted([c for c in df.columns if 'form_' in c or 'rest_' in c or 'congestion' in c])}")


if __name__ == "__main__":
    main()

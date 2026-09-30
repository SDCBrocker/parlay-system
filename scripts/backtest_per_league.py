"""
backtest_per_league.py — Backtest per liga (versi binary 1X2 + ensemble).
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES


def load_models():
    models = {}
    for name in ["model_1x2_home", "model_1x2_away", "model_ou", "model_btts"]:
        p = path(f"models/{name}.pkl")
        if p.exists():
            models[name] = joblib.load(p)
    return models


def _predict_proba(model, X):
    """Handle ensemble dict atau model tunggal."""
    if isinstance(model, dict):
        xgb = model["xgb"].predict_proba(X)
        lgbm = model["lgbm"].predict_proba(X)
        w = model.get("weights", {"xgb": 0.5, "lgbm": 0.5})
        return w["xgb"] * xgb + w["lgbm"] * lgbm
    return model.predict_proba(X)


def evaluate_league(df, models, liga):
    """Evaluasi 1 liga."""
    cols = [c for c in FEATURES if c in df.columns]
    if not cols:
        return None

    X = df[cols].astype(float)

    # 1X2 (pakai binary home/away)
    prob_home = _predict_proba(models["model_1x2_home"], X)[:, 1]
    prob_away = _predict_proba(models["model_1x2_away"], X)[:, 1]

    # Prediksi: home kalau prob_home > prob_away & prob_home > 0.5
    pred_1x2 = np.where(
        (prob_home > 0.5) & (prob_home > prob_away), 0,   # Home Win
        np.where(prob_away > 0.5, 2, 1)                    # Away / Draw
    )
    acc_1x2 = (pred_1x2 == df["result_enc"]).mean() if "result_enc" in df else 0

    # O/U
    prob_ou = _predict_proba(models["model_ou"], X)
    pred_ou = (prob_ou[:, 1] > 0.5).astype(int)
    acc_ou = (pred_ou == df["over_2_5"]).mean() if "over_2_5" in df else 0

    # BTTS
    prob_btts = _predict_proba(models["model_btts"], X)
    pred_btts = (prob_btts[:, 1] > 0.5).astype(int)
    acc_btts = (pred_btts == df["btts"]).mean() if "btts" in df else 0

    # Simulasi value 1X2 Home
    profit = 0
    n_bets = 0
    if "odds_home" in df.columns and "result_enc" in df.columns:
        mask = (pred_1x2 == 0) & df["odds_home"].notna() & (df["odds_home"] > 1)
        sub = df[mask].copy()
        if len(sub) > 0:
            sub["prob_H"] = prob_home[mask]
            sub["value"] = sub["prob_H"] * sub["odds_home"] - 1
            sub = sub[sub["value"] > 0.05]
            if len(sub) > 0:
                won = (sub["result_enc"] == 0).astype(int)
                profits = won * (sub["odds_home"] - 1) - (1 - won)
                profit = profits.sum()
                n_bets = len(sub)

    return {
        "liga": liga, "n": len(df),
        "acc_1x2": round(acc_1x2, 4),
        "acc_ou": round(acc_ou, 4),
        "acc_btts": round(acc_btts, 4),
        "n_bets": n_bets,
        "profit_unit": round(profit, 2),
        "roi": round(profit / n_bets, 4) if n_bets > 0 else 0,
    }


def main():
    print("=== BACKTEST PER LIGA ===")

    models = load_models()
    if len(models) < 4:
        print("[ERROR] Model belum lengkap:", list(models.keys()))
        return

    test_path = path("data/processed/matches_test.csv")
    if not test_path.exists():
        print(f"[ERROR] Test set tidak ada: {test_path}")
        return

    df = pd.read_csv(test_path, low_memory=False)

    if "result_enc" not in df.columns and "result" in df.columns:
        df["result_enc"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    target_leagues = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
                     "Eredivisie", "PrimeiraLiga", "Championship",
                     "Belgian", "Turkish", "Greek", "Swiss"]

    results = []
    for liga in target_leagues:
        liga_df = df[df["league"] == liga]
        if liga_df.empty:
            continue
        res = evaluate_league(liga_df, models, liga)
        if res:
            results.append(res)

    if not results:
        print("Tidak ada hasil")
        return

    df_res = pd.DataFrame(results).sort_values("roi", ascending=False)
    print()
    print(df_res.to_string(index=False))

    out = path("logs/backtest_per_league.txt")
    with open(out, "w") as f:
        f.write("=== BACKTEST PER LIGA ===\n\n")
        f.write(df_res.to_string(index=False))
    print(f"\n[OK] Disimpan: {out}")


if __name__ == "__main__":
    main()

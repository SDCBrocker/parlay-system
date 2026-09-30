"""
backtest_per_season.py — Backtest per musim untuk deteksi data drift.
Input : models/*.pkl + matches_test.csv
Output: logs/backtest_per_season.txt
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
    for name in ["model_1x2", "model_ou", "model_btts"]:
        p = path(f"models/{name}.pkl")
        if p.exists():
            models[name] = joblib.load(p)
    return models


def predict_ensemble(model_tuple, X, task="1x2"):
    """Prediksi pakai ensemble."""
    if isinstance(model_tuple, dict):
        xgb_prob = model_tuple["xgb"].predict_proba(X)
        lgbm_prob = model_tuple["lgbm"].predict_proba(X)
        prob = (xgb_prob + lgbm_prob) / 2
    else:
        prob = model_tuple.predict_proba(X)

    if task == "1x2":
        pred = np.argmax(prob, axis=1)
    else:
        pred = (prob[:, 1] > 0.5).astype(int)
    return pred, prob


def evaluate_season(df, models, season):
    """Evaluasi 1 musim."""
    cols = [c for c in FEATURES if c in df.columns]
    if not cols:
        return None
    X = df[cols].astype(float)

    # 1X2
    pred_1x2, prob_1x2 = predict_ensemble(models["model_1x2"], X, "1x2")
    acc_1x2 = (pred_1x2 == df["result_enc"]).mean() if "result_enc" in df else 0

    # O/U
    pred_ou, _ = predict_ensemble(models["model_ou"], X, "binary")
    acc_ou = (pred_ou == df["over_2_5"]).mean() if "over_2_5" in df else 0

    # BTTS
    pred_btts, _ = predict_ensemble(models["model_btts"], X, "binary")
    acc_btts = (pred_btts == df["btts"]).mean() if "btts" in df else 0

    # Simulasi value 1X2 Home
    profit = 0
    n_bets = 0
    if "odds_home" in df.columns:
        mask = (pred_1x2 == 0) & df["odds_home"].notna() & (df["odds_home"] > 1)
        sub = df[mask].copy()
        if len(sub) > 0:
            sub_prob = prob_1x2[mask][:, 0]
            sub["prob_H"] = sub_prob
            sub["value"] = sub["prob_H"] * sub["odds_home"] - 1
            sub = sub[sub["value"] > 0.05]
            if len(sub) > 0:
                won = (sub["result_enc"] == 0).astype(int)
                profits = won * (sub["odds_home"] - 1) - (1 - won)
                profit = profits.sum()
                n_bets = len(sub)

    return {
        "season": season,
        "n": len(df),
        "acc_1x2": round(acc_1x2, 4),
        "acc_ou": round(acc_ou, 4),
        "acc_btts": round(acc_btts, 4),
        "n_bets": n_bets,
        "profit_unit": round(profit, 2),
        "roi": round(profit / n_bets, 4) if n_bets > 0 else 0,
    }


def main():
    print("=== BACKTEST PER MUSIM ===")

    models = load_models()
    if len(models) < 3:
        print("[ERROR] Model belum lengkap")
        return

    test_path = path("data/processed/matches_test.csv")
    if not test_path.exists():
        print(f"[ERROR] Test set tidak ada: {test_path}")
        return

    df = pd.read_csv(test_path, low_memory=False)

    if "result_enc" not in df.columns and "result" in df.columns:
        df["result_enc"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    if "season" not in df.columns:
        print("[ERROR] Kolom 'season' tidak ada")
        return

    results = []
    for season in sorted(df["season"].astype(str).unique()):
        season_df = df[df["season"].astype(str) == season]
        if season_df.empty:
            continue
        res = evaluate_season(season_df, models, season)
        if res:
            results.append(res)

    if not results:
        print("Tidak ada hasil")
        return

    df_res = pd.DataFrame(results).sort_values("season")

    print()
    print(df_res.to_string(index=False))

    out = path("logs/backtest_per_season.txt")
    with open(out, "w") as f:
        f.write("=== BACKTEST PER MUSIM ===\n\n")
        f.write(df_res.to_string(index=False))
    print(f"\n[OK] Disimpan: {out}")


if __name__ == "__main__":
    main()

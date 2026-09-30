"""
walk_forward.py — Walk-forward validation.
Train beberapa periode, test periode berikutnya.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES, XGB_PARAMS, LGBM_PARAMS


def train_and_eval(train_df, test_df, label_col, name):
    cols = [c for c in FEATURES if c in train_df.columns]

    X_train = train_df[cols].astype(float)
    y_train = train_df[label_col].astype(int)
    X_test = test_df[cols].astype(float)
    y_test = test_df[label_col].astype(int)

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)

    lgbm = LGBMClassifier(**LGBM_PARAMS, class_weight="balanced")
    lgbm.fit(X_train, y_train)
    lgbm_prob = lgbm.predict_proba(X_test)

    ens_prob = 0.5 * xgb_prob + 0.5 * lgbm_prob
    ens_pred = (ens_prob[:, 1] > 0.5).astype(int)

    acc = accuracy_score(y_test, ens_pred)
    return acc, len(X_test)


def main():
    print("=== WALK-FORWARD VALIDATION ===")

    df = pd.read_csv(path("data/processed/matches_normalized.csv"), low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df["is_home_win"] = (df["result"] == "H").astype(int)
    df["is_away_win"] = (df["result"] == "A").astype(int)

    # Walk-forward: train 3 tahun, test 1 tahun
    results = []
    test_years = [2022, 2023, 2024]

    for ty in test_years:
        train_end = f"{ty - 1}-12-31"
        test_start = f"{ty}-01-01"
        test_end = f"{ty}-12-31"

        train_df = df[df["date"] <= train_end]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(train_df) < 5000 or len(test_df) < 100:
            continue

        acc_home, n_test = train_and_eval(train_df, test_df, "is_home_win", "home")
        acc_away, _ = train_and_eval(train_df, test_df, "is_away_win", "away")
        acc_ou, _ = train_and_eval(train_df, test_df, "over_2_5", "ou")
        acc_btts, _ = train_and_eval(train_df, test_df, "btts", "btts")

        results.append({
            "test_year": ty,
            "n_train": len(train_df),
            "n_test": n_test,
            "acc_home": round(acc_home, 4),
            "acc_away": round(acc_away, 4),
            "acc_ou": round(acc_ou, 4),
            "acc_btts": round(acc_btts, 4),
        })

    df_res = pd.DataFrame(results)
    print()
    print(df_res.to_string(index=False))
    print()

    if not df_res.empty:
        print("Rata-rata:")
        for col in ["acc_home", "acc_away", "acc_ou", "acc_btts"]:
            print(f"  {col}: {df_res[col].mean():.4f} (±{df_res[col].std():.4f})")

    out = path("logs/walk_forward.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("=== WALK-FORWARD VALIDATION ===\n\n")
        f.write(df_res.to_string(index=False))
    print(f"\n[OK] Disimpan: {out}")


if __name__ == "__main__":
    main()

"""
train_ou_multi.py — Latih 3 model O/U terpisah (1.5, 2.5, 3.5).
"""
import sys
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES, XGB_PARAMS, LGBM_PARAMS


def train_ou_line(df_train, df_test, label_col, name):
    cols = [c for c in FEATURES if c in df_train.columns]
    sub_train = df_train[cols + [label_col]].dropna()
    sub_test = df_test[cols + [label_col]].dropna()

    X_train = sub_train[cols].astype(float)
    y_train = sub_train[label_col].astype(int)
    X_test = sub_test[cols].astype(float)
    y_test = sub_test[label_col].astype(int)

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train)

    lgbm = LGBMClassifier(**LGBM_PARAMS, class_weight="balanced")
    lgbm.fit(X_train, y_train)

    ens_prob = 0.5 * xgb.predict_proba(X_test) + 0.5 * lgbm.predict_proba(X_test)
    ens_pred = (ens_prob[:, 1] > 0.5).astype(int)

    acc = accuracy_score(y_test, ens_pred)
    print(f"  {name}: acc={acc:.4f}  n_test={len(X_test)}")

    return {"xgb": xgb, "lgbm": lgbm}, acc


def main():
    print("=== TRAINING MULTI-MODEL O/U ===")

    df = pd.read_csv(path("data/processed/matches_normalized.csv"), low_memory=False)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    results = {}
    for line, label in [(1.5, "over_1_5"), (2.5, "over_2_5"), (3.5, "over_3_5")]:
        if label not in df.columns:
            print(f"  [SKIP] {label} tidak ada")
            continue
        name = f"model_ou_{int(line*10)}"
        model, acc = train_ou_line(train_df, test_df, label, name)
        out = path(f"models/{name}.pkl")
        joblib.dump(model, out)
        print(f"    ✓ Disimpan: {out}")
        results[name] = acc

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k}: acc={v:.4f}")


if __name__ == "__main__":
    main()

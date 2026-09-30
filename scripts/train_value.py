"""
train_value.py — Latih model prediksi VALUE (bukan hasil).
Label: value > 0.05 (biner).
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES, XGB_PARAMS


def main():
    print("=== TRAINING MODEL VALUE ===")

    df = pd.read_csv(path("data/processed/matches_normalized.csv"), low_memory=False)
    df = df.sort_values("date").reset_index(drop=True)

    # Hitung value untuk 1X2 Home (pakai odds_home)
    if "odds_home" not in df.columns or "result" not in df.columns:
        print("[ERROR] Kolom odds_home / result tidak ada")
        return

    # Placeholder prob_H — pakai model lama
    import joblib
    model_home_path = path("models/model_1x2_home.pkl")
    if not model_home_path.exists():
        print("[ERROR] Model model_1x2_home.pkl tidak ada. Jalankan train.py dulu.")
        return

    model = joblib.load(model_home_path)
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].astype(float)
    prob_H = 0.5 * model["xgb"].predict_proba(X)[:, 1] + 0.5 * model["lgbm"].predict_proba(X)[:, 1]

    df["prob_H"] = prob_H
    df["value"] = df["prob_H"] * df["odds_home"] - 1
    df["is_value"] = (df["value"] > 0.05).astype(int)

    # Split
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[cols].astype(float)
    y_train = train_df["is_value"]
    X_test = test_df[cols].astype(float)
    y_test = test_df["is_value"]

    # Train
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

    # Simpan
    import joblib
    out = path("models/model_value.pkl")
    joblib.dump(model, out)
    print(f"[OK] Disimpan: {out}")


if __name__ == "__main__":
    main()

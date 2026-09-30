"""
train.py — Latih model dengan binary 1X2 + class weight + ensemble bobot.
Versi 2:
- 1X2 dipecah jadi 2 model binary (Home Win, Away Win)
- Class weight untuk handle imbalance
- Ensemble bobot (XGB + LGBM)
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
from sklearn.metrics import accuracy_score, log_loss, classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


FEATURES = [
    "home_form_5", "away_form_5",
    "home_form_10", "away_form_10",
    "home_goals_for_avg_5", "away_goals_for_avg_5",
    "home_goals_against_avg_5", "away_goals_against_avg_5",
    "home_win_rate", "away_win_rate",
    "h2h_avg_goals", "h2h_home_wins",
    "is_home",
    # Fitur konsistensi
    "home_form_std_5", "away_form_std_5",
    "home_clean_sheet_rate_5", "away_clean_sheet_rate_5",
    "home_failed_score_rate_5", "away_failed_score_rate_5",
    # Fitur rest
    "home_rest_days", "away_rest_days",
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# Bobot ensemble per target
ENSEMBLE_WEIGHTS = {
    "1x2_home": {"xgb": 0.6, "lgbm": 0.4},
    "1x2_away": {"xgb": 0.6, "lgbm": 0.4},
    "ou":       {"xgb": 0.5, "lgbm": 0.5},
    "btts":     {"xgb": 0.7, "lgbm": 0.3},
}


def load_data():
    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        raise FileNotFoundError(f"Input tidak ada: {p}")
    return pd.read_csv(p, low_memory=False)


def split_data(df, test_size=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def prepare_xy(df, label_col):
    cols = [c for c in FEATURES if c in df.columns]
    sub = df[cols + [label_col]].dropna()
    X = sub[cols].astype(float)
    y = sub[label_col].astype(int)
    return X, y, cols


def train_ensemble(X_train, y_train, X_test, y_test, name, task="binary"):
    """Latih XGB + LGBM, gabung dengan bobot."""
    print(f"\n  ── {name} ──")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Label distribusi: {y_train.value_counts().to_dict()}")

    weights = ENSEMBLE_WEIGHTS.get(name, {"xgb": 0.5, "lgbm": 0.5})

    # XGBoost dengan class weight
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=1)
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)

    # LightGBM dengan class weight
    lgbm = LGBMClassifier(**LGBM_PARAMS, class_weight="balanced")
    lgbm.fit(X_train, y_train)
    lgbm_prob = lgbm.predict_proba(X_test)

    # Ensemble dengan bobot
    ens_prob = weights["xgb"] * xgb_prob + weights["lgbm"] * lgbm_prob
    ens_pred = np.argmax(ens_prob, axis=1) if task == "multi" else (ens_prob[:, 1] > 0.5).astype(int)

    # Evaluasi
    try:
        acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
        acc_lgbm = accuracy_score(y_test, lgbm.predict(X_test))
        acc_ens = accuracy_score(y_test, ens_pred)
        ll_ens = log_loss(y_test, ens_prob)
    except Exception:
        acc_xgb = acc_lgbm = acc_ens = ll_ens = float("nan")

    print(f"  XGBoost : acc={acc_xgb:.4f}")
    print(f"  LightGBM: acc={acc_lgbm:.4f}")
    print(f"  Ensemble: acc={acc_ens:.4f}  logloss={ll_ens:.4f}")

    if acc_ens > 0.75:
        print(f"  ⚠⚠ AKURASI >75% — CURIGA LEAKAGE!")

    return {"xgb": xgb, "lgbm": lgbm, "weights": weights}, acc_ens, ll_ens


def backup_old_model(name):
    old = path(f"models/{name}.pkl")
    if old.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = path("models/backup")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, backup_dir / f"{name}_{ts}.pkl")


def save_model(model, name):
    backup_old_model(name)
    out = path(f"models/{name}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"  ✓ Disimpan: {out}")


def main():
    print("=== TRAINING VERSI 2 (Binary 1X2 + Class Weight + Ensemble Bobot) ===")

    df = load_data()
    print(f"Data: {len(df)} baris")
    print(f"Rentang: {df['date'].min()} → {df['date'].max()}")

    train_df, test_df = split_data(df, test_size=0.2)
    print(f"Split: train={len(train_df)}, test={len(test_df)}")

    test_df.to_csv(path("data/processed/matches_test.csv"), index=False)

    results = {}

    # === 1X2 BINARY HOME ===
    for d in [train_df, test_df]:
        d["result_enc"] = d["result"].map({"H": 0, "D": 1, "A": 2})
        d["is_home_win"] = (d["result"] == "H").astype(int)
        d["is_away_win"] = (d["result"] == "A").astype(int)

    X_train, y_train, _ = prepare_xy(train_df, "is_home_win")
    X_test, y_test, _ = prepare_xy(test_df, "is_home_win")
    model_home, acc_h, ll_h = train_ensemble(X_train, y_train, X_test, y_test, "1x2_home")
    save_model(model_home, "model_1x2_home")
    results["1x2_home"] = {"acc": acc_h, "logloss": ll_h}

    # === 1X2 BINARY AWAY ===
    X_train, y_train, _ = prepare_xy(train_df, "is_away_win")
    X_test, y_test, _ = prepare_xy(test_df, "is_away_win")
    model_away, acc_a, ll_a = train_ensemble(X_train, y_train, X_test, y_test, "1x2_away")
    save_model(model_away, "model_1x2_away")
    results["1x2_away"] = {"acc": acc_a, "logloss": ll_a}

    # O/U
    X_train, y_train, _ = prepare_xy(train_df, "over_2_5")
    X_test, y_test, _ = prepare_xy(test_df, "over_2_5")
    model_ou, acc_ou, ll_ou = train_ensemble(X_train, y_train, X_test, y_test, "ou")
    save_model(model_ou, "model_ou")
    results["ou"] = {"acc": acc_ou, "logloss": ll_ou}

    # BTTS
    X_train, y_train, _ = prepare_xy(train_df, "btts")
    X_test, y_test, _ = prepare_xy(test_df, "btts")
    model_btts, acc_btts, ll_btts = train_ensemble(X_train, y_train, X_test, y_test, "btts")
    save_model(model_btts, "model_btts")
    results["btts"] = {"acc": acc_btts, "logloss": ll_btts}

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:10s}: acc={v['acc']:.4f}  logloss={v['logloss']:.4f}")


if __name__ == "__main__":
    main()

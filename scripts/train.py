"""
train.py — Latih model dengan binary 1X2 + class weight + ensemble bobot + versioning + NORMALISASI FIX.
Versi 5:
- 1X2 dipecah jadi 2 model binary (Home Win, Away Win)
- Class weight untuk handle imbalance
- Ensemble bobot (XGB + LGBM)
- Cross-validation untuk deteksi overfitting
- Model metadata logging dengan version tag
- Safety checks untuk data quality
- FIX: Normalisasi dilakukan dari TRAINING SET SAJA
- FIX: Normalisasi parameters disimpan & diaplikasikan ke test set
"""
import sys
import shutil
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path, MODEL_VERSION, N_LEAGUES, TRAINING_DATE, CV_OVERFITTING_THRESHOLD


FEATURES = [
    "home_form_5", "away_form_5",
    "home_form_10", "away_form_10",
    "home_goals_for_avg_5", "away_goals_for_avg_5",
    "home_goals_against_avg_5", "away_goals_against_avg_5",
    "home_win_rate", "away_win_rate",
    "h2h_avg_goals", "h2h_home_wins",
    "is_home",
    "home_form_std_5", "away_form_std_5",
    "home_clean_sheet_rate_5", "away_clean_sheet_rate_5",
    "home_failed_score_rate_5", "away_failed_score_rate_5",
    "home_rest_days", "away_rest_days",
    "home_congestion_7d", "away_congestion_7d",
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

ENSEMBLE_WEIGHTS = {
    "1x2_home": {"xgb": 0.6, "lgbm": 0.4},
    "1x2_away": {"xgb": 0.6, "lgbm": 0.4},
    "ou":       {"xgb": 0.5, "lgbm": 0.5},
    "btts":     {"xgb": 0.7, "lgbm": 0.3},
}


def load_data():
    p = path("data/processed/matches_features.csv")
    if not p.exists():
        raise FileNotFoundError(f"Input tidak ada: {p}")
    df = pd.read_csv(p, low_memory=False)
    print(f"  Loaded: {len(df)} matches")
    return df


def split_data_timeseries(df, test_size=0.2):
    """Split chronologically menggunakan TimeSeriesSplit concept."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def normalize_features(X_train, X_test, feature_cols):
    """
    FIX #1.2: Normalisasi dari TRAINING SET saja.
    - Fit scaler pada X_train
    - Transform X_train dan X_test dengan scaler yang sama
    - Return scaled data dan scaler (untuk predict.py nanti)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_xy(df, label_col):
    """Prepare X, y with robust error handling."""
    cols = [c for c in FEATURES if c in df.columns]
    
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")
    
    if not cols:
        raise ValueError(f"Tidak ada features yang valid untuk {label_col}")
    
    sub = df[cols + [label_col]].dropna()
    if len(sub) == 0:
        raise ValueError(f"Semua baris dropped untuk {label_col}")
    
    X = sub[cols].astype(float)
    y = sub[label_col].astype(int)
    
    # Check class balance
    vc = y.value_counts()
    print(f"    Class distribution: {dict(vc)}")
    
    return X, y, cols


def cross_validate_model(model_class, X, y, cv_folds=5):
    """Hitung CV score dengan TimeSeriesSplit untuk time series data."""
    try:
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = cross_val_score(model_class, X, y, cv=tscv, scoring="accuracy")
        return scores.mean(), scores.std()
    except Exception as e:
        print(f"    CV error: {e}")
        return float("nan"), float("nan")


def train_ensemble(X_train, y_train, X_test, y_test, name, task="binary"):
    """Latih XGB + LGBM, gabung dengan bobot."""
    print(f"\n  ── {name} ──")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

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
    except Exception as e:
        print(f"    Eval error: {e}")
        acc_xgb = acc_lgbm = acc_ens = ll_ens = float("nan")

    print(f"  XGBoost : acc={acc_xgb:.4f}")
    print(f"  LightGBM: acc={acc_lgbm:.4f}")
    print(f"  Ensemble: acc={acc_ens:.4f}  logloss={ll_ens:.4f}")

    # Cross-validation check
    cv_mean, cv_std = cross_validate_model(xgb, X_train, y_train, cv_folds=5)
    print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Overfitting detection
    overfitting_warning = False
    if not np.isnan(cv_mean):
        gap = acc_ens - cv_mean
        if gap > CV_OVERFITTING_THRESHOLD:
            print(f"  ⚠️  POTENTIAL OVERFITTING: gap={gap:.4f} (> threshold {CV_OVERFITTING_THRESHOLD})")
            overfitting_warning = True
        elif acc_ens > 0.78:
            print(f"  ⚠️  HIGH ACCURACY ({acc_ens:.4f}) — verify data leakage")

    return {
        "xgb": xgb, 
        "lgbm": lgbm, 
        "weights": weights,
        "features": list(X_train.columns),
    }, acc_ens, ll_ens, cv_mean, overfitting_warning


def backup_old_model(name):
    """Backup model lama sebelum override."""
    old = path(f"models/{name}.pkl")
    if old.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = path("models/backup")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{name}_{ts}.pkl"
        shutil.copy2(old, backup_path)
        print(f"  Backed up: {backup_path}")


def save_model(model, name, metadata=None):
    """Save model + metadata dengan versioning."""
    backup_old_model(name)
    
    out = path(f"models/{name}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"  ✓ Model saved: {out}")
    
    if metadata:
        meta_out = path(f"models/{name}_meta.json")
        with open(meta_out, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  ✓ Metadata: {meta_out}")


def save_scaler(scaler, name):
    """Save normalization scaler untuk prediction time."""
    out = path(f"models/{name}_scaler.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out)
    print(f"  ✓ Scaler saved: {out}")


def main():
    print("=" * 70)
    print("TRAINING VERSI 5 (Normalisasi FIX + TimeSeriesSplit + Consistency Features)")
    print("=" * 70)
    print(f"Model Version: {MODEL_VERSION}")
    print(f"N Leagues: {N_LEAGUES}")
    print()

    df = load_data()
    print(f"Data range: {df['date'].min()} → {df['date'].max()}")
    print(f"Unique leagues: {df['league'].nunique()}")

    train_df, test_df = split_data_timeseries(df, test_size=0.2)
    print(f"Split: train={len(train_df)}, test={len(test_df)}\n")

    # Save test set for validation
    test_df.to_csv(path("data/processed/matches_test.csv"), index=False)

    results = {}
    all_metadata = {
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "n_leagues": N_LEAGUES,
        "trained_at": datetime.now().isoformat(),
        "normalization": "StandardScaler fitted on training set only",
        "validation_strategy": "TimeSeriesSplit (chronological split)",
        "models": {}
    }

    # === 1X2 BINARY HOME ===
    print("=" * 70)
    print("Training 1X2 HOME")
    print("=" * 70)
    
    for d in [train_df, test_df]:
        d["result_enc"] = d["result"].map({"H": 0, "D": 1, "A": 2})
        d["is_home_win"] = (d["result"] == "H").astype(int)
        d["is_away_win"] = (d["result"] == "A").astype(int)

    X_train, y_train, cols = prepare_xy(train_df, "is_home_win")
    X_test, y_test, _ = prepare_xy(test_df, "is_home_win")
    
    # FIX #1.2: Normalize dari training set saja
    print("  [FIX #1.2] Normalizing features from training set...")
    X_train, X_test, scaler_home = normalize_features(X_train, X_test, cols)
    print("  ✓ Normalization applied (scaler fitted on training set)")
    
    model_home, acc_h, ll_h, cv_h, overfit_h = train_ensemble(X_train, y_train, X_test, y_test, "1x2_home")
    
    meta_h = {
        "name": "model_1x2_home",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": float(acc_h),
        "test_logloss": float(ll_h),
        "cv_mean_accuracy": float(cv_h),
        "overfitting_warning": bool(overfit_h),
        "n_features": len(cols),
        "features": cols,
        "ensemble_weights": model_home["weights"],
    }
    save_model(model_home, "model_1x2_home", meta_h)
    save_scaler(scaler_home, "model_1x2_home")
    all_metadata["models"]["1x2_home"] = meta_h
    results["1x2_home"] = {"acc": acc_h, "logloss": ll_h}

    # === 1X2 BINARY AWAY ===
    print("\n" + "=" * 70)
    print("Training 1X2 AWAY")
    print("=" * 70)
    
    X_train, y_train, cols = prepare_xy(train_df, "is_away_win")
    X_test, y_test, _ = prepare_xy(test_df, "is_away_win")
    
    print("  [FIX #1.2] Normalizing features from training set...")
    X_train, X_test, scaler_away = normalize_features(X_train, X_test, cols)
    print("  ✓ Normalization applied (scaler fitted on training set)")
    
    model_away, acc_a, ll_a, cv_a, overfit_a = train_ensemble(X_train, y_train, X_test, y_test, "1x2_away")
    
    meta_a = {
        "name": "model_1x2_away",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": float(acc_a),
        "test_logloss": float(ll_a),
        "cv_mean_accuracy": float(cv_a),
        "overfitting_warning": bool(overfit_a),
        "n_features": len(cols),
        "features": cols,
        "ensemble_weights": model_away["weights"],
    }
    save_model(model_away, "model_1x2_away", meta_a)
    save_scaler(scaler_away, "model_1x2_away")
    all_metadata["models"]["1x2_away"] = meta_a
    results["1x2_away"] = {"acc": acc_a, "logloss": ll_a}

    # === O/U ===
    print("\n" + "=" * 70)
    print("Training O/U (Over/Under 2.5)")
    print("=" * 70)
    
    X_train, y_train, cols = prepare_xy(train_df, "over_2_5")
    X_test, y_test, _ = prepare_xy(test_df, "over_2_5")
    
    print("  [FIX #1.2] Normalizing features from training set...")
    X_train, X_test, scaler_ou = normalize_features(X_train, X_test, cols)
    print("  ✓ Normalization applied (scaler fitted on training set)")
    
    model_ou, acc_ou, ll_ou, cv_ou, overfit_ou = train_ensemble(X_train, y_train, X_test, y_test, "ou")
    
    meta_ou = {
        "name": "model_ou",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": float(acc_ou),
        "test_logloss": float(ll_ou),
        "cv_mean_accuracy": float(cv_ou),
        "overfitting_warning": bool(overfit_ou),
        "n_features": len(cols),
        "features": cols,
        "ensemble_weights": model_ou["weights"],
    }
    save_model(model_ou, "model_ou", meta_ou)
    save_scaler(scaler_ou, "model_ou")
    all_metadata["models"]["ou"] = meta_ou
    results["ou"] = {"acc": acc_ou, "logloss": ll_ou}

    # === BTTS ===
    print("\n" + "=" * 70)
    print("Training BTTS (Both Teams To Score)")
    print("=" * 70)
    
    X_train, y_train, cols = prepare_xy(train_df, "btts")
    X_test, y_test, _ = prepare_xy(test_df, "btts")
    
    print("  [FIX #1.2] Normalizing features from training set...")
    X_train, X_test, scaler_btts = normalize_features(X_train, X_test, cols)
    print("  ✓ Normalization applied (scaler fitted on training set)")
    
    model_btts, acc_btts, ll_btts, cv_btts, overfit_btts = train_ensemble(X_train, y_train, X_test, y_test, "btts")
    
    meta_btts = {
        "name": "model_btts",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": float(acc_btts),
        "test_logloss": float(ll_btts),
        "cv_mean_accuracy": float(cv_btts),
        "overfitting_warning": bool(overfit_btts),
        "n_features": len(cols),
        "features": cols,
        "ensemble_weights": model_btts["weights"],
    }
    save_model(model_btts, "model_btts", meta_btts)
    save_scaler(scaler_btts, "model_btts")
    all_metadata["models"]["btts"] = meta_btts
    results["btts"] = {"acc": acc_btts, "logloss": ll_btts}

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:10s}: acc={v['acc']:.4f}  logloss={v['logloss']:.4f}")

    # Save master training log
    log_p = path("logs/training_log.json")
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    print(f"\n✓ Master training log: {log_p}")
    print(f"✓ Model version: {MODEL_VERSION}")
    print(f"✓ Normalization scalers saved for prediction stage")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

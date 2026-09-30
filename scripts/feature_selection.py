"""
feature_selection.py — Pilih fitur terbaik berdasarkan importance.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES


def main():
    print("=== FEATURE SELECTION ===")

    # Load model
    p = path("models/model_1x2_home.pkl")
    if not p.exists():
        print("[ERROR] Model tidak ada. Jalankan train.py dulu.")
        return

    model = joblib.load(p)

    # Gabung importance dari xgb + lgbm
    imp_xgb = model["xgb"].feature_importances_
    imp_lgbm = model["lgbm"].feature_importances_
    imp_avg = (imp_xgb + imp_lgbm) / 2

    df_imp = pd.DataFrame({
        "feature": FEATURES[:len(imp_avg)],
        "importance": imp_avg,
    }).sort_values("importance", ascending=False)

    print()
    print(df_imp.to_string(index=False))

    # Fitur yang importance < 0.01
    low = df_imp[df_imp["importance"] < 0.01]
    print()
    print(f"Fitur importance < 0.01: {len(low)}")
    for _, r in low.iterrows():
        print(f"  {r['feature']}: {r['importance']:.4f}")

    # Simpan
    out = path("logs/feature_importance.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out, index=False)
    print(f"\n[OK] Disimpan: {out}")


if __name__ == "__main__":
    main()

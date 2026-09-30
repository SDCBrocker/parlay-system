"""
backtest.py — Backtest 3 model.
Versi 4: bankroll simulation pakai stake flat Rp 10.000 (realistis).
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from train import FEATURES


def load_models():
    models = {}
    for name in ["model_1x2", "model_ou", "model_btts"]:
        p = path(f"models/{name}.pkl")
        if not p.exists():
            raise FileNotFoundError(f"Model tidak ada: {p}")
        models[name] = joblib.load(p)
    return models


def load_test_data() -> pd.DataFrame:
    p = path("data/processed/matches_test.csv")
    if not p.exists():
        raise FileNotFoundError(f"Test set tidak ada: {p}")
    df = pd.read_csv(p, low_memory=False)
    if "result_enc" not in df.columns:
        if "result" in df.columns:
            df["result_enc"] = df["result"].map({"H": 0, "D": 1, "A": 2})
        else:
            raise KeyError("Kolom result/result_enc tidak ada")
    return df


def predict_all(models, df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].astype(float)

    prob_1x2 = models["model_1x2"].predict_proba(X)
    df["prob_H"] = prob_1x2[:, 0]
    df["pred_1x2"] = np.argmax(prob_1x2, axis=1)
    df["conf_1x2"] = np.max(prob_1x2, axis=1)

    prob_ou = models["model_ou"].predict_proba(X)
    df["prob_over_2_5"] = prob_ou[:, 1]
    df["pred_ou"] = (prob_ou[:, 1] > 0.5).astype(int)
    df["conf_ou"] = np.max(prob_ou, axis=1)

    prob_btts = models["model_btts"].predict_proba(X)
    df["pred_btts"] = (prob_btts[:, 1] > 0.5).astype(int)
    df["conf_btts"] = np.max(prob_btts, axis=1)

    return df


def accuracy_per_confidence(df, label_col, pred_col, conf_col):
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    rows = []
    for c in [label_col, pred_col, conf_col]:
        if c not in df.columns:
            return pd.DataFrame()
    correct = (df[label_col] == df[pred_col]).astype(int)
    for lo, hi in bands:
        mask = (df[conf_col] >= lo) & (df[conf_col] < hi)
        n = mask.sum()
        if n > 0:
            rows.append({
                "band": f"{lo:.0%}-{hi:.0%}",
                "n": int(n),
                "accuracy": round(float(correct[mask].mean()), 4),
            })
    return pd.DataFrame(rows)


def value_betting_simulation(df):
    results = {}
    if "odds_home" in df.columns and "result_enc" in df.columns:
        mask = (df["pred_1x2"] == 0) & (df["odds_home"].notna()) & (df["odds_home"] > 1)
        sub = df[mask].copy()
        if len(sub) > 0:
            sub["value"] = sub["prob_H"] * sub["odds_home"] - 1
            sub = sub[sub["value"] > 0.05]
            if len(sub) > 0:
                won = (sub["result_enc"] == 0).astype(int)
                profit = won * (sub["odds_home"] - 1) - (1 - won)
                results["1X2_Home"] = {
                    "n_bets": int(len(sub)),
                    "win_rate": round(float(won.mean()), 4),
                    "roi": round(float(profit.mean()), 4),
                    "total_profit_unit": round(float(profit.sum()), 2),
                }

    if "odds_over_2_5" in df.columns and "over_2_5" in df.columns:
        mask = (df["pred_ou"] == 1) & (df["odds_over_2_5"].notna()) & (df["odds_over_2_5"] > 1)
        sub = df[mask].copy()
        if len(sub) > 0:
            sub["value"] = sub["prob_over_2_5"] * sub["odds_over_2_5"] - 1
            sub = sub[sub["value"] > 0.05]
            if len(sub) > 0:
                won = (sub["over_2_5"] == 1).astype(int)
                profit = won * (sub["odds_over_2_5"] - 1) - (1 - won)
                results["O/U_2.5_Over"] = {
                    "n_bets": int(len(sub)),
                    "win_rate": round(float(won.mean()), 4),
                    "roi": round(float(profit.mean()), 4),
                    "total_profit_unit": round(float(profit.sum()), 2),
                }
    return results


def bankroll_simulation(df, start=1_000_000, stake_flat=10_000, n_legs=3):
    """
    Versi 3: stake FLAT (bukan % dari bankroll).
    1 parlay per hari, filter odds realistis, batas max odds parlay 10.0.
    """
    df = df.sort_values("date").reset_index(drop=True)

    if "odds_home" not in df.columns:
        return {"start": start, "final": start, "roi": 0, "max_drawdown": 0,
                "sharpe": 0, "n_periods": 0}, pd.DataFrame()

    # Filter odds realistis per leg (1.2 - 5.0)
    valid = df["odds_home"].between(1.2, 5.0)
    df = df[valid].copy()
    if df.empty:
        return {"start": start, "final": start, "roi": 0, "max_drawdown": 0,
                "sharpe": 0, "n_periods": 0}, pd.DataFrame()

    bankroll = start
    history = [{"date": df["date"].iloc[0], "bankroll": bankroll}]
    n_parlay = 0

    for date, group in df.groupby("date"):
        if len(group) < n_legs:
            continue

        top = group.nlargest(n_legs, "conf_1x2")

        # Filter: total odds parlay max 10.0 (realistis)
        parlay_odds = top["odds_home"].prod()
        if parlay_odds > 10.0:
            continue

        # Stake FLAT (bukan % dari bankroll)
        stake = min(stake_flat, bankroll)
        if stake <= 0:
            break

        # Cek apakah semua leg menang
        wins = (top["pred_1x2"] == top["result_enc"]).all()

        if wins:
            bankroll += stake * (parlay_odds - 1)
        else:
            bankroll -= stake

        if bankroll <= 0:
            bankroll = 0
            break

        n_parlay += 1
        history.append({"date": date, "bankroll": bankroll})

    hist_df = pd.DataFrame(history)
    final = bankroll
    roi = (final - start) / start if start > 0 else 0
    peak = hist_df["bankroll"].cummax()
    dd = (hist_df["bankroll"] - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0
    returns = hist_df["bankroll"].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    return {
        "start": start, "final": round(final, 2), "roi": round(roi, 4),
        "max_drawdown": round(max_dd, 4), "sharpe": round(sharpe, 4),
        "n_periods": n_parlay,
    }, hist_df


def plot_results(hist_df, acc_df):
    logs = path("logs")
    logs.mkdir(parents=True, exist_ok=True)

    if not hist_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(hist_df["date"]), hist_df["bankroll"])
        plt.title("Bankroll Simulation (Stake Flat Rp 10.000)")
        plt.xlabel("Date")
        plt.ylabel("Bankroll (IDR)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(logs / "backtest_bankroll.png", dpi=100)
        plt.close()

    if not acc_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(acc_df["band"], acc_df["accuracy"])
        plt.title("Accuracy per Confidence Band")
        plt.xlabel("Confidence Band")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(logs / "backtest_accuracy.png", dpi=100)
        plt.close()


def main():
    print("=== BACKTEST ===")

    models = load_models()
    df = load_test_data()
    print(f"Test set: {len(df)} baris")

    df = predict_all(models, df)

    print("\n── Akurasi per confidence band ──")
    acc_1x2 = accuracy_per_confidence(df, "result_enc", "pred_1x2", "conf_1x2")
    acc_ou = accuracy_per_confidence(df, "over_2_5", "pred_ou", "conf_ou")
    acc_btts = accuracy_per_confidence(df, "btts", "pred_btts", "conf_btts")

    print("\n1X2:")
    print(acc_1x2.to_string(index=False) if not acc_1x2.empty else "  (kosong)")
    print("\nO/U 2.5:")
    print(acc_ou.to_string(index=False) if not acc_ou.empty else "  (kosong)")
    print("\nBTTS:")
    print(acc_btts.to_string(index=False) if not acc_btts.empty else "  (kosong)")

    print("\n── Value Betting Simulation ──")
    vb = value_betting_simulation(df)
    for k, v in vb.items():
        print(f"  {k}: {v}")

    print("\n── Bankroll Simulation (stake FLAT Rp 10.000, parlay 3 leg) ──")
    bk, hist_df = bankroll_simulation(df, start=1_000_000, stake_flat=10_000, n_legs=3)
    for k, v in bk.items():
        print(f"  {k}: {v}")

    plot_results(hist_df, acc_1x2)
    print("\n✓ Grafik disimpan di logs/")

    report_path = path("logs/backtest_report.txt")
    with open(report_path, "w") as f:
        f.write("=== BACKTEST REPORT ===\n\n")
        f.write(f"Test set: {len(df)} baris\n\n")
        f.write("1X2:\n" + (acc_1x2.to_string(index=False) if not acc_1x2.empty else "N/A") + "\n\n")
        f.write("O/U 2.5:\n" + (acc_ou.to_string(index=False) if not acc_ou.empty else "N/A") + "\n\n")
        f.write("BTTS:\n" + (acc_btts.to_string(index=False) if not acc_btts.empty else "N/A") + "\n\n")
        f.write("Value Betting:\n")
        for k, v in vb.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nBankroll:\n")
        for k, v in bk.items():
            f.write(f"  {k}: {v}\n")
    print(f"✓ Report: {report_path}")


if __name__ == "__main__":
    main()

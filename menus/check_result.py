"""
menus/check_result.py — Cek hasil laga yang sudah diprediksi.
"""
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import path
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def load_predictions() -> dict:
    p = path("data/processed/predictions_today.json")
    if not p.exists():
        print(f"{Fore.YELLOW}Belum ada predictions_today.json. Jalankan predict.py dulu.{Style.RESET_ALL}")
        return {}
    return json.loads(p.read_text())


def load_results() -> pd.DataFrame:
    """Load hasil aktual dari matches_normalized.csv."""
    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    return df


def check_one_prediction(pred: dict, results: pd.DataFrame) -> dict:
    """Cocokkan 1 prediksi dengan hasil aktual."""
    mid = pred["match_id"]
    row = results[results["match_id"] == mid]
    if row.empty:
        return {"status": "belum", "hasil": "-", "profit": 0}

    r = row.iloc[0]
    market = pred["market"]
    prediction = pred["prediction"]

    # Cek hasil
    if market == "1X2":
        if r["home_goals"] > r["away_goals"]:
            actual = "Home Win"
        elif r["home_goals"] < r["away_goals"]:
            actual = "Away Win"
        else:
            actual = "Draw"
    elif market.startswith("O/U"):
        line = float(market.split()[1])
        total = r["home_goals"] + r["away_goals"]
        actual = "Over" if total > line else "Under"
    elif market == "BTTS":
        actual = "Yes" if (r["home_goals"] > 0 and r["away_goals"] > 0) else "No"
    else:
        return {"status": "skip", "hasil": "-", "profit": 0}

    correct = (actual == prediction)
    stake = pred.get("stake", 10000)
    odds = pred.get("odds", 1.0)
    profit = stake * (odds - 1) if correct else -stake

    return {
        "status": "✅" if correct else "❌",
        "hasil": actual,
        "profit": profit,
    }


def main():
    print(f"{Fore.CYAN}=== CEK HASIL ==={Style.RESET_ALL}")

    preds = load_predictions()
    if not preds:
        return

    results = load_results()
    if results.empty:
        print(f"{Fore.RED}Database hasil tidak ada.{Style.RESET_ALL}")
        return

    all_items = preds.get("tier_s", []) + preds.get("tier_a", []) + preds.get("tier_b", [])
    if not all_items:
        print(f"{Fore.YELLOW}Tidak ada prediksi.{Style.RESET_ALL}")
        return

    rows = []
    total_profit = 0
    n_correct = 0
    n_total = 0

    for p in all_items:
        check = check_one_prediction(p, results)
        if check["status"] in ("✅", "❌"):
            n_total += 1
            if check["status"] == "✅":
                n_correct += 1
            total_profit += check["profit"]
        rows.append({
            "Match": f"{p['home']} vs {p['away']}",
            "Market": p["market"],
            "Prediksi": p["prediction"],
            "Hasil": check["hasil"],
            "Status": check["status"],
            "Profit": f"Rp {check['profit']:+,}" if check["profit"] != 0 else "-",
        })

    print()
    print(tabulate(rows, headers="keys", tablefmt="grid"))

    acc = n_correct / n_total if n_total > 0 else 0
    print()
    print(f"{Fore.GREEN}Akurasi: {n_correct}/{n_total} = {acc:.2%}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total profit: Rp {total_profit:+,}{Style.RESET_ALL}")

    # Log
    log_p = path(f"logs/check_result_{datetime.now().strftime('%Y%m%d')}.log")
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "w") as f:
        f.write(f"Cek hasil {datetime.now()}\n")
        f.write(f"Akurasi: {acc:.2%}\n")
        f.write(f"Profit: Rp {total_profit:+,}\n\n")
        f.write(tabulate(rows, headers="keys", tablefmt="grid"))
    print(f"{Fore.CYAN}Log: {log_p}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

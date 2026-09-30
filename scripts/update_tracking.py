"""
update_tracking.py — Update tracking pending dengan hasil aktual.
Input : data/processed/tracking_pending.csv + matches_normalized.csv
Output: data/processed/tracking_history.csv

Cara pakai:
  python scripts/update_tracking.py
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


def load_pending() -> pd.DataFrame:
    p = path("data/processed/tracking_pending.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_history_db() -> pd.DataFrame:
    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def cek_hasil(pred: dict, db: pd.DataFrame) -> dict:
    """Cocokkan 1 prediksi dengan hasil aktual."""
    mid = pred.get("match_id")
    row = db[db["match_id"] == mid]
    if row.empty:
        return {"status": "pending", "hasil": None, "profit": 0}

    r = row.iloc[0]
    hg = r.get("home_goals")
    ag = r.get("away_goals")

    if pd.isna(hg) or pd.isna(ag):
        return {"status": "pending", "hasil": None, "profit": 0}

    market = pred.get("market", "")
    prediction = pred.get("prediksi", "")
    odds = float(pred.get("odds", 1.0))
    stake = float(pred.get("stake_saya") or pred.get("stake_rekomendasi") or 10000)

    # Tentukan hasil aktual
    if market == "1X2":
        if hg > ag:
            actual = "Home Win"
        elif hg < ag:
            actual = "Away Win"
        else:
            actual = "Draw"
    elif market.startswith("O/U"):
        try:
            line = float(market.split()[1])
        except Exception:
            return {"status": "skip", "hasil": None, "profit": 0}
        total = hg + ag
        actual = "Over" if total > line else "Under"
    elif market == "BTTS":
        actual = "Yes" if (hg > 0 and ag > 0) else "No"
    else:
        return {"status": "skip", "hasil": None, "profit": 0}

    correct = (str(actual).lower() == str(prediction).lower())
    profit = stake * (odds - 1) if correct else -stake

    return {
        "status": "win" if correct else "lose",
        "hasil": actual,
        "profit": round(profit, 2),
    }


def main():
    print("=== UPDATE TRACKING ===")

    pending = load_pending()
    if pending.empty:
        print("Tidak ada tracking_pending.csv")
        return

    db = load_history_db()
    if db.empty:
        print("Database historis kosong")
        return

    print(f"Pending: {len(pending)} baris")
    print(f"Database: {len(db)} baris")

    # Update setiap baris
    updated = []
    n_win = 0
    n_lose = 0
    n_pending = 0
    total_profit = 0

    for _, row in pending.iterrows():
        pred = row.to_dict()
        res = cek_hasil(pred, db)

        row["hasil"] = res["hasil"] if res["hasil"] else row.get("hasil", "")
        row["profit"] = res["profit"] if res["status"] in ("win", "lose") else row.get("profit", 0)

        if res["status"] == "win":
            n_win += 1
            total_profit += res["profit"]
        elif res["status"] == "lose":
            n_lose += 1
            total_profit += res["profit"]
        elif res["status"] == "pending":
            n_pending += 1

        updated.append(row)

    df_updated = pd.DataFrame(updated)

    # Pisahkan yang sudah selesai vs pending
    selesai = df_updated[df_updated["hasil"].notna() & (df_updated["hasil"] != "")]
    pending_baru = df_updated[df_updated["hasil"].isna() | (df_updated["hasil"] == "")]

    print(f"\nHasil:")
    print(f"  Win   : {n_win}")
    print(f"  Lose  : {n_lose}")
    print(f"  Pending: {n_pending}")
    print(f"  Total profit: Rp {total_profit:+,}")

    # Simpan history
    if not selesai.empty:
        hist_path = path("data/processed/tracking_history.csv")
        if hist_path.exists():
            hist_lama = pd.read_csv(hist_path)
            hist_all = pd.concat([hist_lama, selesai], ignore_index=True)
            hist_all = hist_all.drop_duplicates(subset=["id"], keep="last")
        else:
            hist_all = selesai
        hist_all.to_csv(hist_path, index=False)
        print(f"\n[OK] History: {hist_path} ({len(hist_all)} baris)")

    # Simpan pending baru
    pending_path = path("data/processed/tracking_pending.csv")
    if not pending_baru.empty:
        pending_baru.to_csv(pending_path, index=False)
        print(f"[OK] Pending: {pending_path} ({len(pending_baru)} baris)")
    else:
        if pending_path.exists():
            pending_path.unlink()
        print(f"[OK] Semua pending sudah selesai, file dihapus")


if __name__ == "__main__":
    main()

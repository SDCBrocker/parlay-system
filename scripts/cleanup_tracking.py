"""
cleanup_tracking.py — Bersihkan tracking_pending.csv.
Pindahkan yang sudah ada hasil ke tracking_history.csv.
Backup dulu sebelum overwrite.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


def main():
    print("=== CLEANUP TRACKING ===")

    pending_path = path("data/processed/tracking_pending.csv")
    history_path = path("data/processed/tracking_history.csv")

    if not pending_path.exists():
        print("[INFO] tracking_pending.csv tidak ada")
        return

    df = pd.read_csv(pending_path)
    print(f"Pending: {len(df)} baris")

    # Filter: yang sudah ada hasil vs belum
    selesai = df[df["hasil"].notna() & (df["hasil"].astype(str).str.strip() != "")]
    pending = df[df["hasil"].isna() | (df["hasil"].astype(str).str.strip() == "")]

    print(f"  Selesai: {len(selesai)}")
    print(f"  Pending: {len(pending)}")

    # 1. Backup pending lama
    if len(df) > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path(f"data/backup/tracking_pending_{ts}.csv")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(backup_path, index=False)
        print(f"  [OK] Backup: {backup_path}")

    # 2. Tambah yang selesai ke history
    if len(selesai) > 0:
        if history_path.exists():
            hist_lama = pd.read_csv(history_path)
            hist_all = pd.concat([hist_lama, selesai], ignore_index=True)
            # Hapus duplikat berdasarkan id
            if "id" in hist_all.columns:
                hist_all = hist_all.drop_duplicates(subset=["id"], keep="last")
        else:
            hist_all = selesai
        hist_all.to_csv(history_path, index=False)
        print(f"  [OK] History: {history_path} ({len(hist_all)} baris)")

    # 3. Simpan pending baru
    if len(pending) > 0:
        pending.to_csv(pending_path, index=False)
        print(f"  [OK] Pending baru: {pending_path} ({len(pending)} baris)")
    else:
        if pending_path.exists():
            pending_path.unlink()
        print(f"  [OK] Semua pending selesai, file dihapus")

    print()
    print("[OK] Cleanup selesai")


if __name__ == "__main__":
    main()

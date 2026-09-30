"""
menus/update_database.py — Update database manual.
"""
import sys
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import path
from utils import load_csv, save_csv, backup_file, merge_database
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def clear():
    print("\033c", end="")


def load_main_db() -> pd.DataFrame:
    return load_csv(path("data/processed/matches_normalized.csv"))


def preview(df: pd.DataFrame, n: int = 10):
    if df.empty:
        print(f"{Fore.YELLOW}Data kosong.{Style.RESET_ALL}")
        return
    print(tabulate(df.head(n), headers="keys", tablefmt="grid", showindex=False))
    print(f"\nTotal: {len(df)} baris")


def update_from_file():
    src = input(f"{Fore.GREEN}Path file CSV: {Style.RESET_ALL}").strip()
    p = Path(src).expanduser()
    if not p.exists():
        print(f"{Fore.RED}File tidak ada: {p}{Style.RESET_ALL}")
        return
    df_new = pd.read_csv(p, low_memory=False)
    commit(df_new, f"file {p.name}")


def update_from_url():
    import requests
    url = input(f"{Fore.GREEN}URL CSV: {Style.RESET_ALL}").strip()
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df_new = pd.read_csv(StringIO(r.text), low_memory=False)
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return
    commit(df_new, f"URL {url[:50]}")


def update_from_manual():
    print(f"{Fore.CYAN}Input manual (ketik 'selesai' untuk stop):{Style.RESET_ALL}")
    rows = []
    print("Kolom: match_id,league,date,home_team,away_team,home_goals,away_goals")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() == "selesai":
            break
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            print(f"{Fore.RED}Minimal 7 kolom{Style.RESET_ALL}")
            continue
        rows.append({
            "match_id": parts[0], "league": parts[1], "date": parts[2],
            "home_team": parts[3], "away_team": parts[4],
            "home_goals": parts[5], "away_goals": parts[6],
        })
    if not rows:
        return
    commit(pd.DataFrame(rows), "input manual")


def update_from_api():
    print(f"{Fore.CYAN}Jalankan scripts/fetch_api.py untuk update dari API.{Style.RESET_ALL}")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_api.py")], cwd=str(ROOT))


def commit(df_new: pd.DataFrame, label: str):
    if df_new.empty:
        print(f"{Fore.YELLOW}Data baru kosong.{Style.RESET_ALL}")
        return

    db_p = path("data/processed/matches_normalized.csv")
    df_main = load_main_db()
    df_merged = merge_database(df_main, df_new)

    added = len(df_merged) - len(df_main)
    print(f"\n{Fore.CYAN}Preview data baru:{Style.RESET_ALL}")
    preview(df_new, 5)
    print(f"\n{Fore.CYAN}Akan menambah ~{added} baris baru.{Style.RESET_ALL}")

    try:
        c = input(f"{Fore.GREEN}Commit? (y/n): {Style.RESET_ALL}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if c != "y":
        print(f"{Fore.YELLOW}Dibatalkan.{Style.RESET_ALL}")
        return

    backup_file(db_p)
    save_csv(df_merged, db_p)
    print(f"{Fore.GREEN}✓ Tersimpan: {db_p} ({len(df_merged)} baris){Style.RESET_ALL}")


def rollback():
    backup_dir = path("data/backup")
    if not backup_dir.exists():
        print(f"{Fore.YELLOW}Belum ada backup.{Style.RESET_ALL}")
        return
    backups = sorted(backup_dir.glob("matches_normalized_*.csv"), reverse=True)
    if not backups:
        print(f"{Fore.YELLOW}Belum ada backup.{Style.RESET_ALL}")
        return
    for i, b in enumerate(backups[:10], 1):
        print(f"  {i}. {b.name}")
    try:
        c = int(input(f"{Fore.GREEN}Pilih nomor: {Style.RESET_ALL}"))
    except (ValueError, EOFError, KeyboardInterrupt):
        return
    if 1 <= c <= len(backups[:10]):
        src = backups[c - 1]
        dst = path("data/processed/matches_normalized.csv")
        shutil.copy2(src, dst)
        print(f"{Fore.GREEN}✓ Rollback dari {src.name}{Style.RESET_ALL}")


def show_history():
    backup_dir = path("data/backup")
    if not backup_dir.exists():
        print(f"{Fore.YELLOW}Belum ada backup.{Style.RESET_ALL}")
        return
    backups = sorted(backup_dir.glob("*.csv"), reverse=True)[:20]
    for b in backups:
        size = b.stat().st_size / 1024
        print(f"  {b.name} ({size:.1f} KB)")


def main():
    while True:
        clear()
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   💾 UPDATE DATABASE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print("1. Update dari file lokal")
        print("2. Update dari URL")
        print("3. Update dari input manual")
        print("4. Update dari API")
        print("5. Lihat preview data baru")
        print("6. Rollback ke backup terakhir")
        print("7. Lihat history backup")
        print("8. Kembali")
        print()

        try:
            p = input(f"{Fore.GREEN}Pilih [1-8]: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if p == "1":
            update_from_file()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "2":
            update_from_url()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "3":
            update_from_manual()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "4":
            update_from_api()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "5":
            preview(load_main_db())
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "6":
            rollback()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "7":
            show_history()
            input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        elif p == "8":
            break


if __name__ == "__main__":
    main()

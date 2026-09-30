"""
menus/fetch_api.py — Menu interaktif fetch data.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def clear():
    print("\033c", end="")


def run_script(script: str, args: list[str] = None):
    """Jalankan script di scripts/."""
    cmd = [sys.executable, str(ROOT / "scripts" / script)] + (args or [])
    subprocess.run(cmd, cwd=str(ROOT))


def show_history_preview():
    """Lihat preview data history."""
    p = ROOT / "data/raw/history/matches_history.csv"
    if not p.exists():
        print(f"{Fore.YELLOW}Belum ada data history. Jalankan opsi 1 dulu.{Style.RESET_ALL}")
        return
    import pandas as pd
    df = pd.read_csv(p, nrows=10)
    print(tabulate(df.head(10), headers="keys", tablefmt="grid", showindex=False))
    print(f"\n{Fore.CYAN}Total: {sum(1 for _ in open(p)) - 1} baris{Style.RESET_ALL}")


def check_quota():
    """Cek quota API dari file lokal."""
    import json
    q = ROOT / "data/raw/quota.json"
    if q.exists():
        data = json.loads(q.read_text())
        print(f"{Fore.CYAN}Quota terakhir: {data}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Belum ada catatan quota.{Style.RESET_ALL}")


def main():
    while True:
        clear()
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   📡 FETCH DATA{Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════{Style.RESET_ALL}")
        print("1. Fetch History (football-data.co.uk)")
        print("2. Fetch API (fixtures hari ini)")
        print("3. Lihat Preview History")
        print("4. Cek Quota API")
        print("5. Update Database (merge)")
        print("6. Kembali")
        print()

        try:
            p = input(f"{Fore.GREEN}Pilih [1-6]: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if p == "1":
            run_script("fetch_history.py")
            input(f"\n{Fore.CYAN}Enter untuk lanjut...{Style.RESET_ALL}")
        elif p == "2":
            run_script("fetch_api.py")
            input(f"\n{Fore.CYAN}Enter untuk lanjut...{Style.RESET_ALL}")
        elif p == "3":
            show_history_preview()
            input(f"\n{Fore.CYAN}Enter untuk lanjut...{Style.RESET_ALL}")
        elif p == "4":
            check_quota()
            input(f"\n{Fore.CYAN}Enter untuk lanjut...{Style.RESET_ALL}")
        elif p == "5":
            print(f"{Fore.YELLOW}Update database: jalankan fetch dulu, lalu merge otomatis.{Style.RESET_ALL}")
            input(f"\n{Fore.CYAN}Enter untuk lanjut...{Style.RESET_ALL}")
        elif p == "6":
            break
        else:
            print(f"{Fore.RED}Pilihan tidak valid.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

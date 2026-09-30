"""
menus/view_log.py — Lihat log.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from colorama import Fore, Style, init

init(autoreset=True)


LOGS = [
    "main.log", "fetch.log", "clean.log", "features.log",
    "train.log", "backtest.log", "predict.log",
    "update.log", "notify.log",
]


def clear():
    print("\033c", end="")


def warna_baris(line: str) -> str:
    low = line.lower()
    if "error" in low or "traceback" in low or "exception" in low:
        return f"{Fore.RED}{line}{Style.RESET_ALL}"
    if "warning" in low or "warn" in low:
        return f"{Fore.YELLOW}{line}{Style.RESET_ALL}"
    if "✓" in line or "success" in low or "ok" in low:
        return f"{Fore.GREEN}{line}{Style.RESET_ALL}"
    return line


def show_log(name: str, n: int = 50, keyword: str = None):
    p = ROOT / "logs" / name
    if not p.exists():
        print(f"{Fore.YELLOW}Log tidak ada: {name}{Style.RESET_ALL}")
        return
    lines = p.read_text(errors="ignore").splitlines()
    if keyword:
        lines = [l for l in lines if keyword.lower() in l.lower()]
    tail = lines[-n:]
    print(f"{Fore.CYAN}=== {name} (terakhir {len(tail)} baris) ==={Style.RESET_ALL}\n")
    for line in tail:
        print(warna_baris(line))


def main():
    while True:
        clear()
        print(f"{Fore.CYAN}═══ LIHAT LOG ═══{Style.RESET_ALL}\n")
        for i, name in enumerate(LOGS, 1):
            p = ROOT / "logs" / name
            mark = "✓" if p.exists() else "·"
            print(f"  {i:2d}. [{mark}] {name}")
        print(f"  {len(LOGS)+1:2d}. Kembali")
        print()

        try:
            c = input(f"{Fore.GREEN}Pilih: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if c == str(len(LOGS) + 1):
            break

        try:
            idx = int(c) - 1
            if 0 <= idx < len(LOGS):
                clear()
                try:
                    kw = input(f"{Fore.GREEN}Search keyword (Enter skip): {Style.RESET_ALL}").strip() or None
                except (EOFError, KeyboardInterrupt):
                    kw = None
                show_log(LOGS[idx], n=50, keyword=kw)
                input(f"\n{Fore.CYAN}Enter...{Style.RESET_ALL}")
        except ValueError:
            pass


if __name__ == "__main__":
    main()

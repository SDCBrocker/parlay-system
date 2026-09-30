"""
menus/main_menu.py — Menu interaktif utama.
Entry point: bash main_menu.sh
Versi 2: tambah status Apify + menu Cek Apify.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from colorama import Fore, Style, init

init(autoreset=True)


def ensure_directories():
    dirs = [
        "data/raw", "data/raw/cache", "data/raw/history",
        "data/processed", "data/backup",
        "models", "logs", "notebooks",
        "scripts", "menus",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def clear():
    print("\033c", end="")


def run_script(script_path: str):
    full = ROOT / script_path
    if not full.exists():
        print(f"{Fore.RED}Script tidak ada: {script_path}{Style.RESET_ALL}")
        input(f"{Fore.CYAN}Enter...{Style.RESET_ALL}")
        return
    try:
        subprocess.run([sys.executable, str(full)], cwd=str(ROOT))
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Dibatalkan.{Style.RESET_ALL}")
    input(f"\n{Fore.CYAN}Enter untuk kembali...{Style.RESET_ALL}")


def get_apify_status_badge() -> str:
    """Ambil status Apify untuk ditampilkan di header."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from cek_apify import cek_status
        status = cek_status()
        s = status["status"]
        msg = status["message"]

        if s == "OK":
            return f"{Fore.GREEN}✓ Apify: {msg}{Style.RESET_ALL}"
        elif s == "WARNING":
            return f"{Fore.YELLOW}⚠ Apify: {msg}{Style.RESET_ALL}"
        elif s == "CRITICAL":
            return f"{Fore.RED}🚨 Apify: {msg}{Style.RESET_ALL}"
        elif s == "NO_TOKEN":
            return f"{Fore.YELLOW}· Apify: Token belum diisi{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}✗ Apify: Error{Style.RESET_ALL}"
    except Exception as e:
        return f"{Fore.YELLOW}· Apify: {str(e)[:40]}{Style.RESET_ALL}"


def print_header():
    print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║   🎯 SISTEM PREDIKSI PARLAY — TERMUX              ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║   XGBoost · 10 Liga · O/U Fleksibel · Tier        ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════╝{Style.RESET_ALL}")

    # Status Apify di header
    badge = get_apify_status_badge()
    print(f"  {badge}")
    print()


def print_menu():
    print(f"{Fore.GREEN}  📡 DATA{Style.RESET_ALL}")
    print("  1.  Fetch History (football-data.co.uk)")
    print("  2.  Fetch API (fixtures + odds + rotasi)")
    print()
    print(f"{Fore.GREEN}  🛠️  PIPELINE{Style.RESET_ALL}")
    print("  3.  Clean Data")
    print("  4.  Feature Engineering")
    print("  5.  Normalisasi")
    print()
    print(f"{Fore.GREEN}  🤖 MODEL{Style.RESET_ALL}")
    print("  6.  Training 3 Model XGBoost")
    print("  7.  Backtest")
    print("  8.  Backtest Per Liga")
    print()
    print(f"{Fore.GREEN}  🔮 PREDIKSI{Style.RESET_ALL}")
    print("  9.  Prediksi Hari Ini")
    print(" 10.  Cek Result")
    print(" 11.  Update Tracking")
    print(" 11b. Cleanup Tracking")
    print(" 12.  Kirim Telegram")
    print(" 13.  Update Database")
    print()
    print(f"{Fore.GREEN}  🏥 MONITORING{Style.RESET_ALL}")
    print(" 14.  Health Check")
    print(" 15.  Lihat Log")
    print(" 16.  Verify Setup")
    print(" 17.  Cek Status Apify")
    print()
    print(f"{Fore.RED} 18.  Keluar{Style.RESET_ALL}")
    print()


def main():
    ensure_directories()

    while True:
        clear()
        print_header()
        print_menu()

        try:
            p = input(f"{Fore.GREEN}Pilih [1-18]: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.YELLOW}Keluar.{Style.RESET_ALL}")
            break

        if p == "1":
            run_script("scripts/fetch_history.py")
        elif p == "2":
            run_script("scripts/fetch_all.py")
        elif p == "3":
            run_script("scripts/clean_data.py")
        elif p == "4":
            run_script("scripts/features.py")
        elif p == "5":
            run_script("scripts/normalize.py")
        elif p == "6":
            run_script("scripts/train.py")
        elif p == "7":
            run_script("scripts/backtest.py")
        elif p == "8":
            run_script("scripts/backtest_per_league.py")
        elif p == "9":
            run_script("scripts/predict.py")
        elif p == "10":
            run_script("menus/check_result.py")
        elif p == "11":
            run_script("scripts/update_tracking.py")
        elif p == "11b":
            run_script("scripts/cleanup_tracking.py")
        elif p == "12":
            run_script("menus/send_telegram.py")
        elif p == "13":
            run_script("menus/update_database.py")
        elif p == "14":
            run_script("menus/healthcheck.py")
        elif p == "15":
            run_script("menus/view_log.py")
        elif p == "16":
            run_script("scripts/verify_setup.py")
        elif p == "17":
            run_script("scripts/cek_apify.py")
        elif p == "18":
            print(f"{Fore.GREEN}Sampai jumpa! 👋{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Pilihan tidak valid.{Style.RESET_ALL}")
            input(f"{Fore.CYAN}Enter...{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

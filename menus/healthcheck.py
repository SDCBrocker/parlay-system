"""
menus/healthcheck.py — Cek kesehatan sistem.
"""
import sys
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def check_file(p: Path, label: str) -> dict:
    exists = p.exists()
    size = f"{p.stat().st_size / 1024:.1f} KB" if exists else "-"
    return {
        "Cek": label,
        "Status": "✓" if exists else "✗",
        "Info": size if exists else "TIDAK ADA",
    }


def check_disk() -> dict:
    try:
        total, used, free = shutil.disk_usage(str(ROOT))
        free_gb = free / (1024 ** 3)
        status = "✓" if free_gb > 1 else "⚠"
        return {
            "Cek": "Disk space",
            "Status": status,
            "Info": f"{free_gb:.2f} GB free",
        }
    except Exception as e:
        return {"Cek": "Disk space", "Status": "✗", "Info": str(e)}


def check_model_age() -> list[dict]:
    rows = []
    for name in ["model_1x2.pkl", "model_ou.pkl", "model_btts.pkl"]:
        p = ROOT / "models" / name
        if not p.exists():
            rows.append({"Cek": f"Umur {name}", "Status": "✗", "Info": "TIDAK ADA"})
            continue
        age_days = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
        status = "✓" if age_days < 90 else "⚠"
        info = f"{age_days} hari"
        if age_days >= 90:
            info += " (RETRAIN!)"
        rows.append({"Cek": f"Umur {name}", "Status": status, "Info": info})
    return rows


def check_last_errors() -> list[dict]:
    rows = []
    logs = ROOT / "logs"
    if not logs.exists():
        return rows
    for log in sorted(logs.glob("*.log"), reverse=True)[:5]:
        try:
            text = log.read_text(errors="ignore")
            n_err = text.lower().count("error") + text.lower().count("traceback")
            n_warn = text.lower().count("warning")
            status = "✓" if n_err == 0 else "⚠"
            rows.append({
                "Cek": f"Log {log.name}",
                "Status": status,
                "Info": f"{n_err} error, {n_warn} warning",
            })
        except Exception:
            pass
    return rows


def main():
    print(f"{Fore.CYAN}═══ HEALTH CHECK ═══{Style.RESET_ALL}\n")

    rows = []

    # File penting
    rows.append(check_file(ROOT / "data/processed/matches_normalized.csv", "matches_normalized.csv"))
    rows.append(check_file(ROOT / "data/processed/matches_test.csv", "matches_test.csv"))
    rows.append(check_file(ROOT / ".env", ".env"))
    rows.append(check_file(ROOT / "config.py", "config.py"))

    # Model
    for name in ["model_1x2.pkl", "model_ou.pkl", "model_btts.pkl"]:
        rows.append(check_file(ROOT / "models" / name, name))

    # Disk
    rows.append(check_disk())

    # Umur model
    rows.extend(check_model_age())

    # Error log
    rows.extend(check_last_errors())

    print(tabulate(rows, headers="keys", tablefmt="grid"))

    # Kesimpulan
    n_fail = sum(1 for r in rows if r["Status"] == "✗")
    n_warn = sum(1 for r in rows if r["Status"] == "⚠")

    print()
    if n_fail == 0 and n_warn == 0:
        print(f"{Fore.GREEN}✓ Sistem sehat.{Style.RESET_ALL}")
    elif n_fail == 0:
        print(f"{Fore.YELLOW}⚠ {n_warn} warning. Cek tabel di atas.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ {n_fail} gagal, {n_warn} warning. Perbaiki dulu.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

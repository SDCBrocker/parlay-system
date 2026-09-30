"""
cek_apify.py — Cek status kredit Apify.
Input : APIFY_TOKEN di .env
Output: status kredit + notifikasi kalau menipis/habis

Dipakai oleh menus/main_menu.py untuk tampilkan status.
"""
import sys
import os
from datetime import datetime
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_API = "https://api.apify.com/v2"

# Free plan Apify: $5/bulan
FREE_PLAN_LIMIT = 5.00

# Threshold notifikasi
THRESHOLD_WARNING = 0.80   # 80% = warning
THRESHOLD_CRITICAL = 0.95  # 95% = critical


def get_usage() -> dict | None:
    """Ambil penggunaan kredit Apify bulan ini."""
    if not APIFY_TOKEN or "your_" in APIFY_TOKEN:
        return None

    url = f"{APIFY_API}/users/me/usage/monthly"
    headers = {"Authorization": f"Bearer {APIFY_TOKEN}"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        return {"error": str(e)}

    if r.status_code == 401:
        return {"error": "401: Token invalid / expired"}
    elif r.status_code == 403:
        return {"error": "403: Akses ditolak"}
    elif r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:100]}"}

    try:
        data = r.json()
    except Exception as e:
        return {"error": f"Parse error: {e}"}

    # Struktur response Apify:
    # {
    #   "data": {
    #     "monthlyUsageCycle": {"startAt": "...", "endAt": "..."},
    #     "totalUsageCreditsUsdAfterVolumeDiscount": 0.12,
    #     "usage": {...}
    #   }
    # }
    inner = data.get("data", {}) or {}
    used = inner.get("totalUsageCreditsUsdAfterVolumeDiscount", 0) or 0

    return {
        "used": float(used),
        "limit": FREE_PLAN_LIMIT,
        "remaining": float(FREE_PLAN_LIMIT) - float(used),
        "percent": float(used) / FREE_PLAN_LIMIT * 100,
        "cycle": inner.get("monthlyUsageCycle", {}),
    }


def get_limits() -> dict | None:
    """Ambil limit akun Apify (informasi tambahan)."""
    if not APIFY_TOKEN or "your_" in APIFY_TOKEN:
        return None

    url = f"{APIFY_API}/users/me/limits"
    headers = {"Authorization": f"Bearer {APIFY_TOKEN}"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return r.json().get("data", {})
    except Exception:
        return None


def cek_status() -> dict:
    """
    Cek status Apify + return dict:
    {
        "status": "OK" | "WARNING" | "CRITICAL" | "ERROR" | "NO_TOKEN",
        "message": str,
        "used": float,
        "limit": float,
        "remaining": float,
        "percent": float,
    }
    """
    if not APIFY_TOKEN or "your_" in APIFY_TOKEN:
        return {
            "status": "NO_TOKEN",
            "message": "APIFY_TOKEN belum diisi di .env",
            "used": 0, "limit": FREE_PLAN_LIMIT,
            "remaining": 0, "percent": 0,
        }

    usage = get_usage()
    if usage is None or "error" in usage:
        err = usage.get("error", "Unknown") if usage else "No response"
        return {
            "status": "ERROR",
            "message": f"Gagal cek Apify: {err}",
            "used": 0, "limit": FREE_PLAN_LIMIT,
            "remaining": 0, "percent": 0,
        }

    used = usage["used"]
    limit = usage["limit"]
    remaining = usage["remaining"]
    percent = usage["percent"]

    if percent >= THRESHOLD_CRITICAL * 100:
        status = "CRITICAL"
        msg = f"KRITIS! Kredit Apify ${used:.2f}/${limit:.2f} ({percent:.0f}%). Sisa ${remaining:.2f}."
    elif percent >= THRESHOLD_WARNING * 100:
        status = "WARNING"
        msg = f"Warning: Kredit Apify ${used:.2f}/${limit:.2f} ({percent:.0f}%). Sisa ${remaining:.2f}."
    else:
        status = "OK"
        msg = f"Aman: ${used:.2f}/${limit:.2f} ({percent:.0f}%). Sisa ${remaining:.2f}."

    return {
        "status": status,
        "message": msg,
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "percent": percent,
    }


def log_status(status: dict):
    """Simpan status ke log."""
    log_path = path("logs/apify_status.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {status['status']}: {status['message']}\n")


def main():
    print("=== CEK STATUS APIFY ===")
    print()

    status = cek_status()
    log_status(status)

    # Print dengan warna
    from colorama import Fore, Style, init
    init(autoreset=True)

    icons = {
        "OK": f"{Fore.GREEN}[OK]{Style.RESET_ALL}",
        "WARNING": f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}",
        "CRITICAL": f"{Fore.RED}[KRITIS]{Style.RESET_ALL}",
        "ERROR": f"{Fore.RED}[ERROR]{Style.RESET_ALL}",
        "NO_TOKEN": f"{Fore.YELLOW}[NO TOKEN]{Style.RESET_ALL}",
    }
    icon = icons.get(status["status"], "[?]")

    print(f"{icon} {status['message']}")
    print()

    if status["status"] == "OK":
        print(f"Kredit terpakai : ${status['used']:.4f}")
        print(f"Kredit limit    : ${status['limit']:.2f}")
        print(f"Kredit sisa     : ${status['remaining']:.4f}")
        print(f"Persentase      : {status['percent']:.2f}%")
        print()
        print(f"{Fore.GREEN}✓ Apify aman dipakai.{Style.RESET_ALL}")
    elif status["status"] == "WARNING":
        print(f"{Fore.YELLOW}⚠ Kredit Apify menipis. Kurangi pemakaian atau tunggu reset bulan depan.{Style.RESET_ALL}")
    elif status["status"] == "CRITICAL":
        print(f"{Fore.RED}🚨 Kredit Apify hampir habis! Siap-siap tidak bisa scrape sampai bulan depan.{Style.RESET_ALL}")
    elif status["status"] == "NO_TOKEN":
        print(f"{Fore.YELLOW}Tambahkan APIFY_TOKEN di .env:{Style.RESET_ALL}")
        print(f"  1. Buka https://console.apify.com/account/integrations")
        print(f"  2. Copy Personal API Token")
        print(f"  3. Tambah: APIFY_TOKEN=apify_api_xxx ke .env")
    elif status["status"] == "ERROR":
        print(f"{Fore.RED}Cek koneksi internet atau token di .env.{Style.RESET_ALL}")

    return status


if __name__ == "__main__":
    main()

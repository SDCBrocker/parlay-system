"""
menus/send_telegram.py — Kirim rekomendasi ke Telegram (manual).
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import path, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from colorama import Fore, Style, init

init(autoreset=True)


def format_message(preds: dict) -> str:
    lines = []
    lines.append("🎯 *PARLAY HARI INI* 🎯")
    lines.append(f"_{datetime.now().strftime('%d %B %Y')}_")
    lines.append("")

    def render_tier(items, title, emoji):
        if not items:
            return
        lines.append(f"{emoji} *{title}*")
        for i, it in enumerate(items, 1):
            lines.append(
                f"{i}. [{it['league']}] {it['home']} vs {it['away']}\n"
                f"   `{it['market']}` → *{it['prediction']}*\n"
                f"   Conf: {it['confidence']:.0%} | Odds: {it['odds']} | Value: {it['value']:+.1%}"
            )
        lines.append("")

    render_tier(preds.get("tier_s", []), "LAPIS S — PREMIUM", "🏆")
    render_tier(preds.get("tier_a", []), "LAPIS A — LOLOS", "✅")
    render_tier(preds.get("tier_b", []), "LAPIS B — HIBURAN", "🎲")

    lines.append("💰 Stake: Rp 10.000 per market")
    lines.append("⚠️ AI hanya patokan. Keputusan di tangan Anda.")

    return "\n".join(lines)


def send_telegram(text: str, retries: int = 3) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"{Fore.RED}TELEGRAM_TOKEN atau CHAT_ID kosong di .env{Style.RESET_ALL}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                return True
            elif r.status_code == 401:
                print(f"{Fore.RED}401: Token salah{Style.RESET_ALL}")
                return False
            elif r.status_code == 400:
                print(f"{Fore.RED}400: {r.text[:200]}{Style.RESET_ALL}")
                return False
            else:
                print(f"{Fore.YELLOW}Attempt {attempt}: {r.status_code}{Style.RESET_ALL}")
        except requests.RequestException as e:
            print(f"{Fore.YELLOW}Attempt {attempt}: {e}{Style.RESET_ALL}")
        time.sleep(2)

    return False


def main():
    print(f"{Fore.CYAN}=== KIRIM TELEGRAM ==={Style.RESET_ALL}")

    p = path("data/processed/predictions_today.json")
    if not p.exists():
        print(f"{Fore.RED}predictions_today.json tidak ada.{Style.RESET_ALL}")
        return

    preds = json.loads(p.read_text())
    text = format_message(preds)

    print()
    print(text)
    print()

    try:
        c = input(f"{Fore.GREEN}Kirim ke Telegram? (y/n): {Style.RESET_ALL}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if c != "y":
        print(f"{Fore.YELLOW}Dibatalkan.{Style.RESET_ALL}")
        return

    ok = send_telegram(text)

    log_p = path("logs/notify.log")
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "a") as f:
        f.write(f"{datetime.now()} | {'OK' if ok else 'FAIL'}\n")

    if ok:
        print(f"{Fore.GREEN}✓ Terkirim{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Gagal{Style.RESET_ALL}")


if __name__ == "__main__":
    main()

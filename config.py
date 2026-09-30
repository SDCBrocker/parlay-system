"""
config.py — Loader konfigurasi global.
Dipakai oleh semua script di scripts/ dan menus/.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Root project = folder tempat config.py berada
ROOT_DIR = Path(__file__).resolve().parent

# Load .env
load_dotenv(ROOT_DIR / ".env")


# === Path Helper ===
def path(*parts) -> Path:
    """Gabung path relatif ke root project."""
    return ROOT_DIR.joinpath(*parts)


# === API ===
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openfootapi.com/v1")

# === Liga & Musim ===
LEAGUES = [x.strip() for x in os.getenv("LEAGUES", "").split(",") if x.strip()]
SEASONS = [x.strip() for x in os.getenv("SEASONS", "").split(",") if x.strip()]

# === O/U Lines ===
OU_LINES = [float(x.strip()) for x in os.getenv("OU_LINES", "1.5,2.5,3.5").split(",") if x.strip()]

# === Filter Dasar ===
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.03"))

# === Tier ===
TIER_S = {
    "conf": float(os.getenv("TIER_S_CONF", "0.72")),
    "value": float(os.getenv("TIER_S_VALUE", "0.10")),
    "skor": float(os.getenv("TIER_S_SKOR", "0.48")),
    "stake": int(os.getenv("STAKE_S", "10000")),
}
TIER_A = {
    "conf": float(os.getenv("TIER_A_CONF", "0.65")),
    "value": float(os.getenv("TIER_A_VALUE", "0.06")),
    "skor": float(os.getenv("TIER_A_SKOR", "0.40")),
    "stake": int(os.getenv("STAKE_A", "10000")),
}
TIER_B = {
    "conf": float(os.getenv("TIER_B_CONF", "0.55")),
    "value": float(os.getenv("TIER_B_VALUE", "0.03")),
    "skor": float(os.getenv("TIER_B_SKOR", "0.30")),
    "stake": int(os.getenv("STAKE_B", "10000")),
}

MAX_DAILY_STAKE = int(os.getenv("MAX_DAILY_STAKE", "100000"))

# === Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === Database ===
DB_PATH = path(os.getenv("DB_PATH", "data/processed/matches_normalized.csv"))
CACHE_DIR = path(os.getenv("CACHE_DIR", "data/raw/cache"))

# === Bobot Skor ===
SKOR_BOBOT_CONF = float(os.getenv("SKOR_BOBOT_CONF", "0.6"))
SKOR_BOBOT_VALUE = float(os.getenv("SKOR_BOBOT_VALUE", "0.4"))

# === Mapping Liga ke Kode football-data.co.uk ===
# Dipakai di fetch_history.py (TAHAP 2)
LEAGUE_CODES = {
    "EPL": "E0",
    "Championship": "E1",
    "LaLiga": "SP1",
    "SerieA": "I1",
    "Bundesliga": "D1",
    "Ligue1": "F1",
    "Eredivisie": "N1",
    "PrimeiraLiga": "P1",
    "LigaMX": "MEX",
    "Brasileirao": "BRA",
}


def hitung_skor(confidence: float, value: float) -> float:
    """Hitung skor komposit dari confidence dan value."""
    return (confidence * SKOR_BOBOT_CONF) + (value * SKOR_BOBOT_VALUE)


def klasifikasi_tier(confidence: float, value: float) -> str:
    """
    Tentukan tier berdasarkan confidence, value, dan skor.
    Return: 'S', 'A', 'B', atau 'X' (tidak lolos).
    """
    skor = hitung_skor(confidence, value)
    if confidence >= TIER_S["conf"] and value >= TIER_S["value"] and skor >= TIER_S["skor"]:
        return "S"
    if confidence >= TIER_A["conf"] and value >= TIER_A["value"] and skor >= TIER_A["skor"]:
        return "A"
    if confidence >= TIER_B["conf"] and value >= TIER_B["value"] and skor >= TIER_B["skor"]:
        return "B"
    return "X"


def stake_untuk_tier(tier: str) -> int:
    """Return stake IDR untuk tier tertentu."""
    return {"S": TIER_S["stake"], "A": TIER_A["stake"], "B": TIER_B["stake"]}.get(tier, 0)


if __name__ == "__main__":
    # Test cepat
    print("=== CONFIG TEST ===")
    print(f"Root      : {ROOT_DIR}")
    print(f"Leagues   : {LEAGUES}")
    print(f"Seasons   : {SEASONS}")
    print(f"O/U Lines : {OU_LINES}")
    print(f"Tier S    : {TIER_S}")
    print(f"Tier A    : {TIER_A}")
    print(f"Tier B    : {TIER_B}")
    print(f"Max stake : {MAX_DAILY_STAKE}")
    print()
    print("Test klasifikasi:")
    for c, v in [(0.80, 0.15), (0.70, 0.08), (0.60, 0.04), (0.50, 0.02)]:
        t = klasifikasi_tier(c, v)
        s = hitung_skor(c, v)
        print(f"  conf={c} value={v} → skor={s:.4f} tier={t}")

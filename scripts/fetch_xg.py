"""
fetch_xg.py — Scrape xG dari Understat.
Cover 6 liga: EPL, LaLiga, SerieA, Bundesliga, Ligue1, RFPL.
Output: data/raw/xg/xg_{league}_{season}.csv
"""
import sys
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


# Understat league codes
UNDERSTAT_LEAGUES = {
    "EPL":        "EPL",
    "LaLiga":     "La_liga",
    "Bundesliga": "Bundesliga",
    "SerieA":     "Serie_A",
    "Ligue1":     "Ligue_1",
    "RFPL":       "RFPL",
}

SEASONS = ["2020", "2021", "2022", "2023", "2024"]


def fetch_league_season(league_code: str, season: str) -> pd.DataFrame | None:
    """Scrape 1 liga 1 musim dari Understat."""
    url = f"https://understat.com/league/{league_code}/{season}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] {e}")
        return None

    # Extract JSON dari HTML
    # Understat simpan data di variabel JavaScript: var datesData = JSON.parse('...')
    match = re.search(r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    if not match:
        print(f"  [ERROR] Tidak bisa parse JSON")
        return None

    # Decode escaped JSON
    raw = match.group(1)
    raw = raw.encode().decode("unicode_escape")
    data = json.loads(raw)

    if not data:
        return None

    rows = []
    for match in data:
        h = match.get("h", {})
        a = match.get("a", {})
        goals = match.get("goals", {})
        xg = match.get("xG", {})

        rows.append({
            "date": match.get("datetime"),
            "home_team": h.get("title"),
            "away_team": a.get("title"),
            "home_goals": int(goals.get("h", 0)) if goals.get("h") else None,
            "away_goals": int(goals.get("a", 0)) if goals.get("a") else None,
            "home_xg": float(xg.get("h", 0)) if xg.get("h") else None,
            "away_xg": float(xg.get("a", 0)) if xg.get("a") else None,
            "league": league_code,
            "season": season,
        })

    return pd.DataFrame(rows)


def main():
    print("=== FETCH xG (Understat) ===")
    out_dir = path("data/raw/xg")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for liga, code in UNDERSTAT_LEAGUES.items():
        for season in SEASONS:
            print(f"  {liga} {season}...", end=" ")
            df = fetch_league_season(code, season)
            if df is not None and not df.empty:
                out = out_dir / f"xg_{liga}_{season}.csv"
                df.to_csv(out, index=False)
                print(f"[OK] {len(df)} baris")
                total += len(df)
            else:
                print("[SKIP]")
            time.sleep(2)  # rate limit sopan

    print(f"\n[OK] Total: {total} baris")
    print(f"[OK] Disimpan di: {out_dir}")


if __name__ == "__main__":
    main()

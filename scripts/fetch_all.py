"""
fetch_all.py — Script master: fixtures + odds + rotasi API.
Versi 2: fuzzy matching untuk nama tim.
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path
from api_rotator import APIKeyRotator
from league_whitelist import filter_fixtures_by_whitelist


LEAGUE_TO_SPORT_KEY = {
    "EPL":          "soccer_epl",
    "LaLiga":       "soccer_spain_la_liga",
    "SerieA":       "soccer_italy_serie_a",
    "Bundesliga":   "soccer_germany_bundesliga",
    "Ligue1":       "soccer_france_ligue_one",
    "Eredivisie":   "soccer_netherlands_eredivisie",
    "PrimeiraLiga": "soccer_portugal_primeira_liga",
    "Championship": "soccer_efl_champ",
    "LigaMX":       "soccer_mexico_ligamx",
    "Brasileirao":  "soccer_brazil_campeonato",
    # Liga baru
    "Belgian":      "soccer_belgium_first_div",
    "Turkish":      "soccer_turkey_super_league",
    "Greek":        "soccer_greece_super_league",
    "Swiss":        "soccer_switzerland_superleague",
}

SKIP_KEYWORDS = [
    "canadian", "canada", "_can", "peru", "_per", "liga_1_per",
    "nigeria", "npfl", "_nga", "colombia", "primera_a_col", "_col",
    "uruguay", "_uru", "usl", "_usa", "championship_usa",
    "mls", "major_league", "mexico", "_mex",
    "japan", "j_league", "_jpn", "china", "csl", "_chn",
    "korea", "k_league", "_kor", "australia", "a_league", "_aus",
    "india", "isl", "_ind", "saudi", "spl", "_sau",
    "qatar", "_qat", "egypt", "_egy", "south_africa", "_rsa",
]

COMPETITION_KEYWORDS = {
    "premier_league_eng": "EPL", "england_premier": "EPL", "_eng_premier": "EPL",
    "epl_2026": "EPL", "epl_2025": "EPL",
    "championship_eng": "Championship", "efl_championship": "Championship",
    "la_liga_esp": "LaLiga", "spain_la_liga": "LaLiga", "laliga": "LaLiga",
    "esl_2026": "LaLiga", "esl_2025": "LaLiga",
    "serie_a_ita": "SerieA", "italy_serie": "SerieA", "seriea": "SerieA",
    "bundesliga_ger": "Bundesliga", "germany_bundesliga": "Bundesliga", "bundesliga": "Bundesliga",
    "ligue_1_fra": "Ligue1", "france_ligue": "Ligue1", "ligue1": "Ligue1",
    "eredivisie_ned": "Eredivisie", "netherlands_eredivisie": "Eredivisie", "eredivisie": "Eredivisie",
    "primeira_liga_por": "PrimeiraLiga", "portugal_primeira": "PrimeiraLiga", "primeira_liga": "PrimeiraLiga",
}

CACHE_DIR = path("data/raw/cache")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower()
    # Buang kata umum
    for w in ["fc", "sc", "cf", "ac", "afc", "club", "de", "the"]:
        n = n.replace(f" {w} ", " ").replace(f" {w}", "").replace(f"{w} ", "")
    # Normalisasi karakter
    n = (n.replace(" ", "").replace(".", "").replace("-", "")
         .replace("_", "").replace("'", "")
         .replace("ü", "u").replace("é", "e").replace("á", "a")
         .replace("í", "i").replace("ó", "o").replace("ñ", "n")
         .replace("ö", "o").replace("ä", "a").replace("ß", "ss")
         .replace("è", "e").replace("ê", "e").replace("ç", "c"))
    return n


def fuzzy_match(name1: str, name2: str, threshold: float = 0.75) -> bool:
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    ratio = SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold


def map_competition(comp_id: str):
    if not comp_id:
        return None
    cl = comp_id.lower()
    for skip in SKIP_KEYWORDS:
        if skip in cl:
            return None
    for kw, kode in COMPETITION_KEYWORDS.items():
        if kw in cl:
            return kode
    return None


def gen_match_id(liga, date, home, away):
    try:
        dt = pd.to_datetime(date)
        dt_str = dt.strftime("%Y%m%d")
    except Exception:
        dt_str = "00000000"
    h = str(home).replace(" ", "").replace(".", "").replace("-", "")
    a = str(away).replace(" ", "").replace(".", "").replace("-", "")
    return f"{liga}_{dt_str}_{h}_{a}"


def parse_openfoot_fixture(item):
    comp_id = item.get("competitionId", "")
    liga = map_competition(comp_id)
    if not liga:
        return None
    home = item.get("homeTeam", {}) or {}
    away = item.get("awayTeam", {}) or {}
    score = item.get("score", {}) or {}
    kickoff = item.get("kickoffAt")
    try:
        date = pd.to_datetime(kickoff).tz_localize(None) if kickoff else pd.NaT
    except Exception:
        date = pd.NaT
    home_name = home.get("name") or home.get("shortName") or ""
    away_name = away.get("name") or away.get("shortName") or ""
    return {
        "match_id": gen_match_id(liga, date, home_name, away_name),
        "date": date,
        "league": liga,
        "season": item.get("season"),
        "home_team": home_name,
        "away_team": away_name,
        "home_goals": score.get("home"),
        "away_goals": score.get("away"),
        "source": "api",
        "_status": item.get("status"),
        "_competition_id": comp_id,
    }


def fetch_fixtures(rotator, date):
    print(f"\n[OPENFOOT] Fetch fixtures: {date}")
    while rotator.has_available():
        key = rotator.get()
        url = "https://openfootapi.com/v1/matches"
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        try:
            r = requests.get(url, headers=headers, params={"date": date}, timeout=30)
        except requests.RequestException as e:
            print(f"  [ERROR] Network: {e}")
            return pd.DataFrame()
        if r.status_code == 200:
            data = r.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            print(f"  [OK] {len(items)} fixture")
            parsed = [parse_openfoot_fixture(it) for it in items]
            parsed = [p for p in parsed if p is not None]
            print(f"  [OK] Setelah filter: {len(parsed)} fixture target")
            return pd.DataFrame(parsed) if parsed else pd.DataFrame()
        elif r.status_code in (401, 429):
            print(f"  [WARN] {r.status_code}: rotate...")
            rotator.mark_exhausted(key)
        else:
            print(f"  [ERROR] {r.status_code}")
            return pd.DataFrame()
    return pd.DataFrame()


def cache_path(sport_key):
    return CACHE_DIR / f"odds_{sport_key}.json"


def load_cache(sport_key, ttl_hours=24):
    p = cache_path(sport_key)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > ttl_hours * 3600:
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save_cache(sport_key, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(sport_key).write_text(json.dumps(data, indent=2))


def fetch_odds_league(rotator, sport_key):
    print(f"\n[ODDS] Fetch {sport_key}")
    cached = load_cache(sport_key)
    if cached is not None:
        print(f"  [CACHE] {len(cached)} events")
        return cached
    while rotator.has_available():
        key = rotator.get()
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {"apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"}
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  [ERROR] Network: {e}")
            return []
        if r.status_code == 200:
            data = r.json()
            save_cache(sport_key, data)
            remaining = r.headers.get("x-requests-remaining", "?")
            print(f"  [OK] {len(data)} events | quota left: {remaining}")
            return data
        elif r.status_code in (401, 429):
            print(f"  [WARN] {r.status_code}: rotate...")
            rotator.mark_exhausted(key)
        elif r.status_code == 422:
            print(f"  [SKIP] {sport_key} tidak tersedia")
            return []
        else:
            print(f"  [ERROR] {r.status_code}")
            return []
    return []


def parse_odds_event(event):
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    oh, od, oa, oo, ou = [], [], [], [], []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            mkey = market.get("key")
            for o in market.get("outcomes", []):
                name = o.get("name", "")
                price = o.get("price")
                if not price:
                    continue
                if mkey == "h2h":
                    if normalize_name(name) == normalize_name(home_team):
                        oh.append(price)
                    elif normalize_name(name) == normalize_name(away_team):
                        oa.append(price)
                    elif name.lower() == "draw":
                        od.append(price)
                elif mkey == "totals":
                    if o.get("point") != 2.5:
                        continue
                    if name.lower() == "over":
                        oo.append(price)
                    elif name.lower() == "under":
                        ou.append(price)
    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None
    return {
        "home_team": home_team,
        "away_team": away_team,
        "odds_home": avg(oh),
        "odds_draw": avg(od),
        "odds_away": avg(oa),
        "odds_over_2_5": avg(oo),
        "odds_under_2_5": avg(ou),
    }


def main():
    print("=" * 60)
    print("  FETCH ALL — Fixtures + Odds (Rotasi API)")
    print("=" * 60)

    openfoot_keys = os.getenv("OPENFOOT_API_KEYS", "") or os.getenv("API_KEY", "")
    odds_keys = os.getenv("ODDS_API_KEYS", "") or os.getenv("ODDS_API_KEY", "")

    if not openfoot_keys or not odds_keys:
        print("[ERROR] API keys belum diisi di .env")
        return

    openfoot_rot = APIKeyRotator(openfoot_keys, "OPENFOOT")
    odds_rot = APIKeyRotator(odds_keys, "ODDS")

    today = datetime.now()
    all_fixtures = []
    for i in range(0, 4):
        date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        df = fetch_fixtures(openfoot_rot, date)
        if not df.empty:
            all_fixtures.append(df)

    if not all_fixtures:
        print("\n[INFO] Tidak ada fixture target 4 hari ke depan.")
        return

    fixtures = pd.concat(all_fixtures, ignore_index=True)
    fixtures = fixtures.drop_duplicates(subset=["match_id"])
    print(f"\n[OK] Total fixtures: {len(fixtures)}")

    leagues = fixtures["league"].unique().tolist()
    valid_leagues = [l for l in leagues if l in LEAGUE_TO_SPORT_KEY]

    all_odds = []
    for liga in valid_leagues:
        sport_key = LEAGUE_TO_SPORT_KEY[liga]
        events = fetch_odds_league(odds_rot, sport_key)
        for ev in events:
            parsed = parse_odds_event(ev)
            parsed["league"] = liga
            all_odds.append(parsed)

    if not all_odds:
        print("\n[WARN] Tidak ada odds.")
        return

    odds_df = pd.DataFrame(all_odds)
    print(f"\n[OK] Total events dengan odds: {len(odds_df)}")

    # Fuzzy matching
    matched = 0
    for i, fix in fixtures.iterrows():
        for _, odd in odds_df.iterrows():
            if odd["league"] != fix["league"]:
                continue
            if fuzzy_match(fix["home_team"], odd["home_team"]) and fuzzy_match(fix["away_team"], odd["away_team"]):
                for col in ["odds_home", "odds_draw", "odds_away", "odds_over_2_5", "odds_under_2_5"]:
                    if pd.notna(odd.get(col)):
                        fixtures.at[i, col] = odd[col]
                matched += 1
                print(f"  [MATCH] {fix['home_team']} vs {fix['away_team']} <-> {odd['home_team']} vs {odd['away_team']}")
                break

    print(f"\n[OK] Matching: {matched}/{len(fixtures)} laga dapat odds")

    out = path("data/raw/fixtures_today.csv")
    df_save = fixtures.drop(columns=["_status", "_competition_id"], errors="ignore")
    df_save.to_csv(out, index=False)
    print(f"[OK] Disimpan: {out}")

    print("\n--- PREVIEW ---")
    cols = ["match_id", "league", "home_team", "away_team",
            "odds_home", "odds_draw", "odds_away", "odds_over_2_5", "odds_under_2_5"]
    cols = [c for c in cols if c in df_save.columns]
    print(df_save[cols].to_string(index=False))


if __name__ == "__main__":
    main()


def load_manual_odds_fallback():
    """
    Fallback: baca odds manual dari data/raw/manual_odds.csv.
    Format: match_id,odds_home,odds_draw,odds_away,...
    """
    manual_path = path("data/raw/manual_odds.csv")
    if not manual_path.exists():
        return {}
    try:
        df = pd.read_csv(manual_path)
        return df.set_index("match_id").to_dict("index")
    except Exception:
        return {}


def apply_manual_odds(fixtures, manual_odds):
    """Override odds fixtures dengan manual kalau ada."""
    if not manual_odds:
        return fixtures
    n_override = 0
    for i, row in fixtures.iterrows():
        mid = row["match_id"]
        if mid in manual_odds:
            mo = manual_odds[mid]
            for col in ["odds_home", "odds_draw", "odds_away",
                        "odds_over_2_5", "odds_under_2_5"]:
                if col in mo and pd.notna(mo[col]):
                    fixtures.at[i, col] = mo[col]
            n_override += 1
    if n_override > 0:
        print(f"  [OK] Manual odds override: {n_override} laga")
    return fixtures

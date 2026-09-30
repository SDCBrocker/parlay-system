"""
fetch_api.py — Wrapper OpenFootAPI.
Versi 4: auto-generate match_id format database + simpan fixtures.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import API_KEY, API_BASE_URL, path
from schema import ensure_schema, derive_labels
from utils import load_cache, save_cache, load_csv, save_csv, backup_file, merge_database


HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
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
    "costa_rica", "_crc", "romania", "_ro", "super_liga_ro",
    "moldova", "_mda", "albania", "_alb", "armenia", "_arm",
    "azerbaijan", "_aze", "belarus", "_blr", "bulgaria", "_bul",
    "croatia", "_cro", "cyprus", "_cyp", "czech", "_cze",
    "denmark", "_den", "estonia", "_est", "finland", "_fin",
    "georgia", "_geo", "greece", "_gre", "hungary", "_hun",
    "iceland", "_isl", "israel", "_isr", "kazakhstan", "_kaz",
    "latvia", "_lva", "lithuania", "_ltu", "luxembourg", "_lux",
    "malta", "_mlt", "norway", "_nor", "poland", "_pol",
    "russia", "_rus", "serbia", "_srb", "slovakia", "_svk",
    "slovenia", "_svn", "sweden", "_swe", "switzerland", "_sui",
    "turkey", "_tur", "ukraine", "_ukr", "wales", "_wal",
    "austria", "_aut", "belgium", "_bel", "scotland", "_sco",
    "ireland", "_irl", "northern_ireland", "_nir",
]

COMPETITION_KEYWORDS = {
    # EPL
    "premier_league_eng": "EPL",
    "england_premier": "EPL",
    "_eng_premier": "EPL",
    "epl_2026": "EPL",
    "epl_2025": "EPL",
    # Championship
    "championship_eng": "Championship",
    "efl_championship": "Championship",
    # LaLiga
    "la_liga_esp": "LaLiga",
    "spain_la_liga": "LaLiga",
    "laliga": "LaLiga",
    "esl_2026": "LaLiga",
    "esl_2025": "LaLiga",
    # SerieA
    "serie_a_ita": "SerieA",
    "italy_serie": "SerieA",
    "seriea": "SerieA",
    # Bundesliga
    "bundesliga_ger": "Bundesliga",
    "germany_bundesliga": "Bundesliga",
    "bundesliga": "Bundesliga",
    # Ligue1
    "ligue_1_fra": "Ligue1",
    "france_ligue": "Ligue1",
    "ligue1": "Ligue1",
    # Eredivisie
    "eredivisie_ned": "Eredivisie",
    "netherlands_eredivisie": "Eredivisie",
    "eredivisie": "Eredivisie",
    # PrimeiraLiga
    "primeira_liga_por": "PrimeiraLiga",
    "portugal_primeira": "PrimeiraLiga",
    "primeira_liga": "PrimeiraLiga",
}


def api_get(endpoint: str, params: dict = None, use_cache: bool = True) -> dict | None:
    params = params or {}
    if use_cache:
        cached = load_cache(endpoint, params)
        if cached is not None:
            print(f"  [CACHE] {endpoint}")
            return cached

    url = f"{API_BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"  [ERROR] Network: {e}")
        return None

    if r.status_code == 200:
        data = r.json()
        if use_cache:
            save_cache(endpoint, params, data)
        return data
    elif r.status_code == 401:
        print("  [ERROR] 401: API_KEY salah / expired")
        return None
    elif r.status_code == 429:
        retry = int(r.headers.get("Retry-After", 60))
        print(f"  [WARN] 429: rate limit. Tunggu {retry}s...")
        time.sleep(retry)
        return api_get(endpoint, params, use_cache=False)
    elif r.status_code == 404:
        print(f"  [WARN] 404: {endpoint}")
        return None
    else:
        print(f"  [ERROR] {r.status_code}: {r.text[:100]}")
        return None


def map_competition(comp_id: str) -> str | None:
    if not comp_id:
        return None
    comp_lower = comp_id.lower()

    for skip in SKIP_KEYWORDS:
        if skip in comp_lower:
            return None

    for keyword, kode in COMPETITION_KEYWORDS.items():
        if keyword in comp_lower:
            return kode

    return None


def gen_match_id(liga: str, date, home: str, away: str) -> str:
    """Generate match_id format database: {league}_{YYYYMMDD}_{home}_{away}"""
    try:
        dt = pd.to_datetime(date)
        dt_str = dt.strftime("%Y%m%d")
    except Exception:
        dt_str = "00000000"
    h = str(home).replace(" ", "").replace(".", "").replace("-", "")
    a = str(away).replace(" ", "").replace(".", "").replace("-", "")
    return f"{liga}_{dt_str}_{h}_{a}"


def parse_fixture(item: dict) -> dict | None:
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

    # Auto-generate match_id format database
    new_mid = gen_match_id(liga, date, home_name, away_name)

    return {
        "match_id": new_mid,
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
        "_api_id": item.get("id"),
    }


def fetch_fixtures(date: str = None, league: str = None) -> pd.DataFrame:
    params = {}
    if date:
        params["date"] = date
    if league:
        params["league"] = league

    data = api_get("/matches", params)
    if not data:
        return pd.DataFrame()

    items = data.get("data", []) if isinstance(data, dict) else []
    if not items:
        return pd.DataFrame()

    print(f"  API mengembalikan {len(items)} fixture")

    parsed = []
    skipped = 0
    for item in items:
        row = parse_fixture(item)
        if row is None:
            skipped += 1
            continue
        parsed.append(row)

    print(f"  Setelah filter liga target: {len(parsed)} fixture (skip {skipped})")

    if not parsed:
        return pd.DataFrame()

    return pd.DataFrame(parsed)


def update_database(df_new: pd.DataFrame) -> pd.DataFrame:
    if df_new.empty:
        return pd.DataFrame()

    df_finished = df_new[
        df_new["_status"].isin(["finished", "FT", "AET", "PEN"]) &
        df_new["home_goals"].notna() &
        df_new["away_goals"].notna()
    ].copy()

    if df_finished.empty:
        print("  [INFO] Tidak ada laga finished untuk training.")
        return pd.DataFrame()

    df_finished = df_finished.drop(columns=["_status", "_competition_id", "_api_id"], errors="ignore")
    df_finished = ensure_schema(df_finished)
    df_finished = derive_labels(df_finished)

    db_path = path("data/processed/matches_normalized.csv")
    df_main = load_csv(db_path)

    backup_file(db_path)
    df_merged = merge_database(df_main, df_finished)
    save_csv(df_merged, db_path)
    print(f"  [OK] Database: {len(df_merged)} baris (+{len(df_finished)})")
    return df_merged


def main():
    print("=== FETCH API (OpenFootAPI) ===")

    if not API_KEY or "your_" in API_KEY:
        print("[ERROR] API_KEY belum diisi di .env")
        return

    # Cek hari ini, kalau kosong cek 3 hari ke depan
    from datetime import timedelta
    today = datetime.now()
    dates_to_check = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, 4)]

    all_fixtures = []
    for date in dates_to_check:
        print(f"\n--- Cek tanggal {date} ---")
        df = fetch_fixtures(date=date)
        if not df.empty:
            all_fixtures.append(df)

    if not all_fixtures:
        print("\n[INFO] Tidak ada fixture target dalam 4 hari ke depan.")
        out_fix = path("data/raw/fixtures_today.csv")
        if out_fix.exists():
            out_fix.unlink()
            print(f"[OK] Fixtures lama dihapus")
        return

    df_all = pd.concat(all_fixtures, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["match_id"])

    out_fix = path("data/raw/fixtures_today.csv")
    out_fix.parent.mkdir(parents=True, exist_ok=True)
    # Simpan tanpa kolom internal untuk fixtures
    df_save = df_all.drop(columns=["_status", "_competition_id", "_api_id"], errors="ignore")
    df_save.to_csv(out_fix, index=False)
    print(f"\n[OK] Fixtures: {out_fix} ({len(df_save)} baris)")

    update_database(df_all)


if __name__ == "__main__":
    main()

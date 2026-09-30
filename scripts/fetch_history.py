"""
fetch_history.py — Scrape CSV historis dari football-data.co.uk.
URL pattern: https://www.football-data.co.uk/mmz4281/<YYSS>/<KODE>.csv
Ref: https://www.football-data.co.uk/notes.txt
"""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import LEAGUES, SEASONS, path
from schema import FD_MAPPING, FD_LEAGUE_CODES, ensure_schema, derive_labels


BASE_URL = "https://www.football-data.co.uk/mmz4281"


def season_code(season: str) -> str:
    """2023 → '2324' (musim 2023/24)."""
    y = int(season)
    return f"{str(y)[-2:]}{str(y + 1)[-2:]}"


def download_csv(league: str, season: str) -> pd.DataFrame | None:
    """Download 1 CSV. Return None kalau 404 / tidak ada."""
    code = FD_LEAGUE_CODES.get(league)
    if not code:
        return None

    sc = season_code(season)
    url = f"{BASE_URL}/{sc}/{code}.csv"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            print(f"  ⚠ {league} {season}: file tidak ada")
            return None
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ {league} {season}: {e}")
        return None

    # Parse CSV dari string
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
    except Exception as e:
        print(f"  ❌ Parse error: {e}")
        return None

    if df.empty:
        return None

    # Tambah metadata sekaligus (hindari fragmentation)
    df = df.assign(league=league, season=season, source="history")

    # Rename sesuai mapping
    df = df.rename(columns=FD_MAPPING)

    # Buang kolom yang tidak perlu
    df = ensure_schema(df)
    df = derive_labels(df)

    return df


def fetch_all(leagues: list[str] = None, seasons: list[str] = None) -> pd.DataFrame:
    """Download semua kombinasi league × season."""
    leagues = leagues or LEAGUES
    seasons = seasons or SEASONS

    # Hanya liga yang ada di FD_LEAGUE_CODES
    valid = [l for l in leagues if l in FD_LEAGUE_CODES]
    skipped = [l for l in leagues if l not in FD_LEAGUE_CODES]

    print(f"Download {len(valid)} liga × {len(seasons)} musim")
    if skipped:
        print(f"⚠ Lewati (tidak ada di football-data): {skipped}")

    frames = []
    for league in valid:
        for season in seasons:
            print(f"  → {league} {season}...", end=" ")
            df = download_csv(league, season)
            if df is not None and not df.empty:
                frames.append(df)
                print(f"✓ {len(df)} baris")
            # rate limit sopan
            import time; time.sleep(1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Buat match_id dari league+date+home+away
    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["match_id"] = (
        combined["league"].astype(str) + "_" +
        combined["date"].dt.strftime("%Y%m%d") + "_" +
        combined["home_team"].astype(str).str.replace(" ", "") + "_" +
        combined["away_team"].astype(str).str.replace(" ", "")
    )

    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def main():
    print("=== FETCH HISTORY (football-data.co.uk) ===")
    df = fetch_all()

    if df.empty:
        print("❌ Tidak ada data.")
        return

    out_dir = path("data/raw/history")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "matches_history.csv"
    df.to_csv(out_file, index=False)

    print(f"\n✓ Total: {len(df)} baris")
    print(f"✓ Disimpan: {out_file}")
    print(f"✓ Rentang: {df['date'].min()} → {df['date'].max()}")
    print(f"✓ Liga: {sorted(df['league'].unique())}")


if __name__ == "__main__":
    main()

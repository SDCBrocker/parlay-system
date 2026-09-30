"""
validate_fixtures.py — Validasi fixtures_today.csv sebelum prediksi.
Dipanggil oleh predict.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


REQUIRED_COLUMNS = [
    "match_id", "league", "date", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away",
]

OPTIONAL_COLUMNS = [
    "odds_over_2_5", "odds_under_2_5",
    "odds_btts_yes", "odds_btts_no",
]


def validate(fixtures_path: Path = None) -> tuple[bool, list[str]]:
    """
    Validasi fixtures_today.csv.
    Return: (is_valid, list_of_errors)
    """
    errors = []

    if fixtures_path is None:
        fixtures_path = path("data/raw/fixtures_today.csv")

    # 1. Cek file ada
    if not fixtures_path.exists():
        return False, [f"File tidak ada: {fixtures_path}"]

    # 2. Cek file tidak kosong
    try:
        df = pd.read_csv(fixtures_path)
    except Exception as e:
        return False, [f"Gagal baca CSV: {e}"]

    if df.empty:
        return False, ["File kosong (0 baris)"]

    # 3. Cek kolom wajib
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Kolom wajib hilang: {missing}")

    # 4. Cek kolom opsional
    missing_opt = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_opt:
        errors.append(f"⚠ Kolom opsional hilang (akan di-skip): {missing_opt}")

    # 5. Cek odds valid (harus > 1)
    for col in ["odds_home", "odds_draw", "odds_away"]:
        if col in df.columns:
            invalid = df[(df[col].notna()) & (df[col] <= 1)]
            if len(invalid) > 0:
                errors.append(f"⚠ {col}: {len(invalid)} baris dengan odds ≤ 1 (akan di-skip)")

    # 6. Cek tim kosong
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            empty = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
            if len(empty) > 0:
                errors.append(f"❌ {col}: {len(empty)} baris dengan tim kosong")

    # 7. Cek liga valid
    valid_leagues = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
                     "Eredivisie", "PrimeiraLiga", "Championship", "LigaMX", "Brasileirao"]
    if "league" in df.columns:
        invalid_lg = df[~df["league"].isin(valid_leagues)]
        if len(invalid_lg) > 0:
            errors.append(f"⚠ {len(invalid_lg)} baris dengan liga tidak dikenal")

    # 8. Cek match_id duplikat
    if "match_id" in df.columns:
        dupes = df[df.duplicated("match_id", keep=False)]
        if len(dupes) > 0:
            errors.append(f"⚠ {len(dupes)} baris duplikat match_id")

    # Filter hanya error (bukan warning)
    real_errors = [e for e in errors if not e.startswith("⚠")]

    return len(real_errors) == 0, errors


def main():
    print("=== VALIDASI FIXTURES ===")
    is_valid, errors = validate()

    if is_valid:
        print("[OK] Fixtures valid")
    else:
        print("[ERROR] Fixtures tidak valid:")
        for e in errors:
            print(f"  - {e}")
        return 1

    if errors:
        print()
        print("Warning:")
        for e in errors:
            print(f"  {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

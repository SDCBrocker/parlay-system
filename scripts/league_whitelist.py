"""
league_whitelist.py — Whitelist tim per liga.
Versi 3: longgarkan — pakai whitelist statis + auto-generate sebagai fallback.
"""
import sys
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


# === WHITELIST STATIS (tim yang jelas ada di liga target 2025/26) ===
STATIC_WHITELIST = {
    "EPL": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
        "Leeds", "Leicester", "Liverpool", "Manchester City", "Manchester United",
        "Newcastle", "Nottingham Forest", "Southampton", "Tottenham",
        "West Ham", "Wolverhampton", "Hull City", "Burnley", "Sheffield United",
        "Luton", "Sunderland", "Middlesbrough", "West Brom", "Norwich",
    ],
    "LaLiga": [
        "Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona",
        "Celta Vigo", "Elche", "Espanyol", "Getafe", "Girona",
        "Las Palmas", "Leganes", "Mallorca", "Osasuna", "Rayo Vallecano",
        "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla",
        "Valencia", "Villarreal", "Almeria", "Cadiz", "Granada",
        "Valladolid", "Levante", "Eibar", "Sporting Gijon",
    ],
    "SerieA": [
        "Atalanta", "Bologna", "Cagliari", "Como", "Empoli",
        "Fiorentina", "Genoa", "Hellas Verona", "Inter", "Juventus",
        "Lazio", "Lecce", "Milan", "Monza", "Napoli",
        "Parma", "Roma", "Torino", "Udinese", "Venezia",
        "Sassuolo", "Salernitana", "Frosinone", "Cremonese",
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum",
        "Borussia Dortmund", "Borussia Mönchengladbach", "Eintracht Frankfurt",
        "Freiburg", "Heidenheim", "Hoffenheim", "Holstein Kiel",
        "Mainz", "RB Leipzig", "St. Pauli", "Stuttgart", "Union Berlin",
        "Werder Bremen", "Wolfsburg", "Darmstadt", "Köln",
        "Schalke", "Hertha BSC", "Hamburger SV", "Fortuna Düsseldorf",
    ],
    "Ligue1": [
        "Angers", "Auxerre", "Brest", "Le Havre", "Lens",
        "Lille", "Lyon", "Marseille", "Monaco", "Montpellier",
        "Nantes", "Nice", "Paris Saint-Germain", "PSG", "Reims",
        "Rennes", "Saint-Etienne", "Strasbourg", "Toulouse",
        "Clermont", "Metz", "Lorient", "Bordeaux",
    ],
    "Eredivisie": [
        "Ajax", "Almere City", "AZ", "Feyenoord", "Fortuna Sittard",
        "Go Ahead Eagles", "Groningen", "Heerenveen", "Heracles",
        "NAC Breda", "NEC", "PEC Zwolle", "PSV", "RKC Waalwijk",
        "Sparta Rotterdam", "Twente", "Utrecht", "Willem II",
        "Excelsior", "Vitesse", "Cambuur", "Emmen",
    ],
    "PrimeiraLiga": [
        "Arouca", "AVS", "Benfica", "Boavista", "Braga",
        "Casa Pia", "Estoril", "Estrela", "Famalicao", "Farense",
        "Gil Vicente", "Moreirense", "Nacional", "Porto",
        "Rio Ave", "Santa Clara", "Sporting", "Vitoria Guimaraes",
        "Sp Lisbon", "Sp Braga",
    ],
    "Championship": [
        "Blackburn", "Bristol City", "Burnley", "Cardiff", "Coventry",
        "Derby", "Hull", "Leeds", "Luton", "Middlesbrough",
        "Millwall", "Norwich", "Oxford", "Plymouth", "Portsmouth",
        "Preston", "QPR", "Sheffield United", "Sheffield Wednesday",
        "Stoke", "Sunderland", "Swansea", "Watford", "West Brom",
    ],
    "Belgian": [
        "Anderlecht", "Club Brugge", "Genk", "Gent", "Standard Liege",
        "Antwerp", "Charleroi", "Cercle Brugge", "Kortrijk", "Mechelen",
        "OH Leuven", "Sint-Truiden", "Union SG", "Westerlo", "Beerschot",
        "Dender", "STVV", "St Truiden",
    ],
    "Turkish": [
        "Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor", "Basaksehir",
        "Adana Demirspor", "Alanyaspor", "Antalyaspor", "Bodrum",
        "Eyupspor", "Gaziantep", "Goztepe", "Hatayspor", "Kasimpasa",
        "Kayserispor", "Konyaspor", "Rizespor", "Samsunspor", "Sivasspor",
    ],
    "Greek": [
        "Olympiacos", "Panathinaikos", "AEK Athens", "PAOK", "Aris",
        "Atromitos", "Asteras Tripolis", "Kallithea", "Lamia",
        "Levadiakos", "OFI", "Panserraikos", "Volos",
    ],
    "Swiss": [
        "Young Boys", "Basel", "Zurich", "Servette", "Lugano",
        "St. Gallen", "Luzern", "Sion", "Grasshopper", "Lausanne",
        "Yverdon", "Winterthur",
    ],
}


def normalize(name: str) -> str:
    if not name:
        return ""
    n = str(name).lower()
    for w in ["fc", "sc", "cf", "ac", "afc", "club", "de", "the", "1.", "05", "04"]:
        n = n.replace(f" {w} ", " ").replace(f" {w}", "").replace(f"{w} ", "")
    n = (n.replace(" ", "").replace(".", "").replace("-", "")
         .replace("_", "").replace("'", "")
         .replace("ü", "u").replace("é", "e").replace("á", "a")
         .replace("í", "i").replace("ó", "o").replace("ñ", "n")
         .replace("ö", "o").replace("ä", "a").replace("ß", "ss"))
    return n


def fuzzy_match(n1: str, n2: str, threshold: float = 0.70) -> bool:
    """Fuzzy matching dengan threshold lebih longgar."""
    if not n1 or not n2:
        return False
    if n1 == n2 or n1 in n2 or n2 in n1:
        return True
    return SequenceMatcher(None, n1, n2).ratio() >= threshold


def is_in_whitelist(team: str, league: str, threshold: float = 0.70) -> bool:
    """Cek apakah tim ada di whitelist liga."""
    wl = STATIC_WHITELIST.get(league)
    if not wl:
        return True  # Liga tidak ada whitelist, terima semua
    tn = normalize(team)
    for wt in wl:
        wn = normalize(wt)
        if tn == wn:
            return True
        if fuzzy_match(tn, wn, threshold):
            return True
    return False


def filter_fixtures_by_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Filter fixtures: kedua tim harus ada di whitelist liga."""
    if df.empty:
        return df

    mask = []
    for _, row in df.iterrows():
        liga = row.get("league", "")
        home = row.get("home_team", "")
        away = row.get("away_team", "")

        home_ok = is_in_whitelist(home, liga)
        away_ok = is_in_whitelist(away, liga)

        if home_ok and away_ok:
            mask.append(True)
        else:
            if not home_ok:
                print(f"  [SKIP] {home} tidak di whitelist {liga}")
            if not away_ok:
                print(f"  [SKIP] {away} tidak di whitelist {liga}")
            mask.append(False)

    return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    print("=== TEST WHITELIST STATIS ===")
    for liga, teams in STATIC_WHITELIST.items():
        print(f"{liga}: {len(teams)} tim")

    print()
    print("=== Test case ===")
    test_cases = [
        ("Elche", "LaLiga"),
        ("Celta Vigo", "LaLiga"),
        ("Nottingham Forest", "EPL"),
        ("Hull City", "EPL"),
        ("Karlsruher SC", "Bundesliga"),
    ]
    for team, liga in test_cases:
        ok = is_in_whitelist(team, liga)
        print(f"  {team} di {liga}: {'✓' if ok else '✗'}")

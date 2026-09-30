"""
predict.py — Prediksi fixtures + filter 3 tier + bobot liga.
Versi 4: bobot skor per liga (bukan filter buang liga).
"""
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    path, OU_LINES, TIER_S, TIER_A, TIER_B,
    MAX_DAILY_STAKE, hitung_skor, klasifikasi_tier, stake_untuk_tier,
)
from train import FEATURES
from validate_fixtures import validate as validate_fixtures


# === Bobot liga berdasarkan ROI backtest per liga ===
# Formula: bobot = 1 + (ROI × 3), dibatasi 0.5 - 1.5
# ROI +10% -> 1.30, ROI -10% -> 0.70, ROI -30% -> 0.50 (clamp)
LEAGUE_WEIGHTS = {
    "Eredivisie":   1.30,   # ROI +9.62%
    "Ligue1":       1.20,   # ROI +6.66%
    "PrimeiraLiga": 0.95,   # ROI -2.55%
    "Championship": 0.90,   # ROI -4.15%
    "LaLiga":       0.70,   # ROI -20.80%
    "SerieA":       0.65,   # ROI -22.57%
    "EPL":          0.60,   # ROI -28.22%
    "Bundesliga":   0.55,   # ROI -32.00%
    # Liga default (tidak ada di backtest)
    "LigaMX":       0.80,
    "Brasileirao":  0.80,
}

# Threshold minimum skor setelah bobot
MIN_WEIGHTED_SCORE = 0.30

FIXTURE_COLUMNS = [
    "match_id", "league", "date", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away",
    "odds_over_2_5", "odds_under_2_5",
    "odds_btts_yes", "odds_btts_no",
]


def load_models():
    models = {}
    for name in ["model_1x2_home", "model_1x2_away", "model_ou", "model_btts"]:
        p = path(f"models/{name}.pkl")
        if not p.exists():
            raise FileNotFoundError(f"Model tidak ada: {p}")
        models[name] = joblib.load(p)
    return models


def load_fixtures() -> pd.DataFrame:
    p = path("data/raw/fixtures_today.csv")
    if not p.exists():
        raise FileNotFoundError(
            f"Fixtures tidak ada: {p}\n"
            f"Buat file CSV dengan kolom: {','.join(FIXTURE_COLUMNS)}"
        )
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Auto-generate match_id kalau kosong
    for i, row in df.iterrows():
        mid = str(row.get("match_id", "")).strip()
        if not mid or mid.lower() in ("nan", "none", ""):
            liga = str(row.get("league", "UNK")).strip()
            dt = pd.to_datetime(row.get("date"), errors="coerce")
            dt_str = dt.strftime("%Y%m%d") if pd.notna(dt) else "00000000"
            home = str(row.get("home_team", "")).replace(" ", "")
            away = str(row.get("away_team", "")).replace(" ", "")
            df.at[i, "match_id"] = f"{liga}_{dt_str}_{home}_{away}"

    return df


def load_historical_stats() -> pd.DataFrame:
    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def get_team_stats(hist: pd.DataFrame, team: str, is_home: int) -> dict:
    if hist.empty:
        return {}
    mask = (hist["home_team"] == team) | (hist["away_team"] == team)
    team_df = hist[mask].sort_values("date").tail(10)
    if team_df.empty:
        return {}
    form_col = "home_form_5" if is_home else "away_form_5"
    form_col10 = "home_form_10" if is_home else "away_form_10"
    gf_col = "home_goals_for_avg_5" if is_home else "away_goals_for_avg_5"
    ga_col = "home_goals_against_avg_5" if is_home else "away_goals_against_avg_5"
    win_col = "home_win_rate" if is_home else "away_win_rate"
    return {
        "form_5": team_df[form_col].mean() if form_col in team_df else 1.5,
        "form_10": team_df[form_col10].mean() if form_col10 in team_df else 1.5,
        "goals_for_avg_5": team_df[gf_col].mean() if gf_col in team_df else 1.5,
        "goals_against_avg_5": team_df[ga_col].mean() if ga_col in team_df else 1.2,
        "win_rate": team_df[win_col].mean() if win_col in team_df else 0.4,
    }


def build_features(fixtures: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if not hist.empty:
        league_avg = hist.groupby("league").agg({
            "home_goals": "mean", "away_goals": "mean",
            "home_win_rate": "mean", "away_win_rate": "mean",
            "h2h_avg_goals": "mean", "h2h_home_wins": "mean",
        }).to_dict("index")
    else:
        league_avg = {}

    rows = []
    for _, f in fixtures.iterrows():
        liga = f.get("league", "EPL")
        avg = league_avg.get(liga, {})
        home = f.get("home_team", "")
        away = f.get("away_team", "")
        home_stats = get_team_stats(hist, home, is_home=1)
        away_stats = get_team_stats(hist, away, is_home=0)
        rows.append({
            "match_id": f.get("match_id"),
            "league": liga,
            "date": f.get("date"),
            "home_team": home,
            "away_team": away,
            "odds_home": f.get("odds_home"),
            "odds_draw": f.get("odds_draw"),
            "odds_away": f.get("odds_away"),
            "odds_over_2_5": f.get("odds_over_2_5"),
            "odds_under_2_5": f.get("odds_under_2_5"),
            "odds_btts_yes": f.get("odds_btts_yes"),
            "odds_btts_no": f.get("odds_btts_no"),
            "home_form_5": home_stats.get("form_5", 1.5),
            "away_form_5": away_stats.get("form_5", 1.2),
            "home_form_10": home_stats.get("form_10", 1.5),
            "away_form_10": away_stats.get("form_10", 1.2),
            "home_goals_for_avg_5": home_stats.get("goals_for_avg_5", avg.get("home_goals", 1.5)),
            "away_goals_for_avg_5": away_stats.get("goals_for_avg_5", avg.get("away_goals", 1.2)),
            "home_goals_against_avg_5": home_stats.get("goals_against_avg_5", avg.get("away_goals", 1.2)),
            "away_goals_against_avg_5": away_stats.get("goals_against_avg_5", avg.get("home_goals", 1.5)),
            "home_win_rate": home_stats.get("win_rate", avg.get("home_win_rate", 0.45)),
            "away_win_rate": away_stats.get("win_rate", avg.get("away_win_rate", 0.30)),
            "h2h_avg_goals": avg.get("h2h_avg_goals", 2.5),
            "h2h_home_wins": avg.get("h2h_home_wins", 2.0),
            "is_home": 1,
        })
    return pd.DataFrame(rows)


def predict_all(models, df: pd.DataFrame) -> list[dict]:
    X = df[[c for c in FEATURES if c in df.columns]].astype(float)
    results = []
    def _predict(model, X):
        if isinstance(model, dict):
            xgb = model["xgb"].predict_proba(X)
            lgbm = model["lgbm"].predict_proba(X)
            return (xgb + lgbm) / 2
        return model.predict_proba(X)

    prob_1x2 = _predict(models["model_1x2"], X)
    prob_ou = _predict(models["model_ou"], X)
    prob_btts = _predict(models["model_btts"], X)

    for i, row in df.iterrows():
        preds = []
        # 1X2
        p_h, p_d, p_a = prob_1x2[i]
        choices = [
            ("Home Win", p_h, row.get("odds_home")),
            ("Draw", p_d, row.get("odds_draw")),
            ("Away Win", p_a, row.get("odds_away")),
        ]
        best = max(choices, key=lambda x: x[1])
        preds.append({
            "market": "1X2", "prediction": best[0],
            "confidence": round(float(best[1]), 4),
            "odds": float(best[2]) if pd.notna(best[2]) else None,
        })

        # O/U
        p_over_2_5 = prob_ou[i][1]
        for line in OU_LINES:
            if line == 2.5:
                p_over = p_over_2_5
            elif line == 1.5:
                p_over = min(0.95, p_over_2_5 + 0.20)
            elif line == 3.5:
                p_over = max(0.05, p_over_2_5 - 0.20)
            else:
                p_over = p_over_2_5
            p_under = 1 - p_over
            if p_over >= p_under:
                preds.append({
                    "market": f"O/U {line}", "prediction": "Over",
                    "confidence": round(float(p_over), 4),
                    "odds": row.get(f"odds_over_{line}") if f"odds_over_{line}" in row else row.get("odds_over_2_5"),
                })
            else:
                preds.append({
                    "market": f"O/U {line}", "prediction": "Under",
                    "confidence": round(float(p_under), 4),
                    "odds": row.get(f"odds_under_{line}") if f"odds_under_{line}" in row else row.get("odds_under_2_5"),
                })

        # BTTS
        p_btts = prob_btts[i][1]
        if p_btts >= 0.5:
            preds.append({
                "market": "BTTS", "prediction": "Yes",
                "confidence": round(float(p_btts), 4),
                "odds": row.get("odds_btts_yes"),
            })
        else:
            preds.append({
                "market": "BTTS", "prediction": "No",
                "confidence": round(float(1 - p_btts), 4),
                "odds": row.get("odds_btts_no"),
            })

        results.append({
            "match_id": row["match_id"], "league": row["league"],
            "home": row["home_team"], "away": row["away_team"],
            "predictions": preds,
        })
    return results


def filter_and_classify(results: list[dict]) -> tuple[list, list, list]:
    tier_s, tier_a, tier_b = [], [], []

    for match in results:
        liga = match["league"]
        bobot = LEAGUE_WEIGHTS.get(liga, 0.80)

        for p in match["predictions"]:
            conf = p["confidence"]
            odds = p["odds"]
            if odds is None or pd.isna(odds) or odds <= 1:
                continue
            value = conf * odds - 1

            # Skor dasar (belum bobot)
            skor_dasar = hitung_skor(conf, value)
            # Skor setelah bobot liga
            skor_final = skor_dasar * bobot

            # Threshold: skor final minimal
            if skor_final < MIN_WEIGHTED_SCORE:
                continue

            # Klasifikasi tier pakai skor final
            tier = klasifikasi_tier(conf, value)
            # Kalau skor_final tinggi, naikkan tier
            if skor_final >= TIER_S["skor"] * 1.1 and conf >= TIER_S["conf"]:
                tier = "S"
            elif skor_final >= TIER_A["skor"] * 1.1 and conf >= TIER_A["conf"]:
                tier = "A"
            elif tier == "X":
                tier = "B"  # Skor final > threshold → masuk B

            item = {
                "match_id": match["match_id"], "league": liga,
                "home": match["home"], "away": match["away"],
                "market": p["market"], "prediction": p["prediction"],
                "confidence": round(conf, 4),
                "odds": round(float(odds), 2),
                "value": round(value, 4),
                "skor_dasar": round(skor_dasar, 4),
                "bobot_liga": bobot,
                "skor": round(skor_final, 4),
                "tier": tier,
                "stake": stake_untuk_tier(tier),
            }

            if tier == "S":
                tier_s.append(item)
            elif tier == "A":
                tier_a.append(item)
            elif tier == "B":
                tier_b.append(item)

    tier_s.sort(key=lambda x: x["skor"], reverse=True)
    tier_a.sort(key=lambda x: x["skor"], reverse=True)
    tier_b.sort(key=lambda x: x["skor"], reverse=True)
    return tier_s, tier_a, tier_b


def render_txt(tier_s, tier_a, tier_b, total_scan: int) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"   PREDIKSI PARLAY — {datetime.now().strftime('%d %B %Y')}")
    lines.append(f"   Total scan: {total_scan} laga")
    lines.append(f"   Lolos seleksi: {len(tier_s) + len(tier_a) + len(tier_b)} market")
    lines.append("=" * 60)
    lines.append("")

    def render_tier(items, title, emoji):
        if not items:
            lines.append(f"{emoji} {title} — (kosong)")
            lines.append("")
            return
        lines.append(f"{emoji} {title} ({len(items)} market)")
        lines.append("-" * 60)
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. [{it['league']}] {it['home']} vs {it['away']}")
            lines.append(f"   Match ID  : {it['match_id']}")
            lines.append(f"   Market    : {it['market']} — {it['prediction']}")
            lines.append(f"   Conf      : {it['confidence']:.0%} | Odds: {it['odds']} | Value: {it['value']:+.2%}")
            lines.append(f"   Skor      : {it['skor']:.4f} (dasar {it['skor_dasar']:.4f} × bobot {it['bobot_liga']:.2f})")
            lines.append(f"   Stake     : Rp {it['stake']:,}")
            lines.append("")

    render_tier(tier_s, "LAPIS S — PREMIUM", "🏆")
    render_tier(tier_a, "LAPIS A — LOLOS", "✅")
    render_tier(tier_b, "LAPIS B — HIBURAN", "🎲")

    total_stake = sum(it["stake"] for it in tier_s + tier_a + tier_b)
    total_stake = min(total_stake, MAX_DAILY_STAKE)

    lines.append("=" * 60)
    lines.append(f"TOTAL STAKE: Rp {total_stake:,} (maks Rp {MAX_DAILY_STAKE:,})")
    lines.append("Catatan: Skor sudah dibobot per liga. AI hanya penyaring awal.")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_tracking(tier_s, tier_a, tier_b):
    all_items = tier_s + tier_a + tier_b
    if not all_items:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for i, it in enumerate(all_items, 1):
        rows.append({
            "id": i, "tanggal": today, "match_id": it["match_id"],
            "liga": it["league"], "home": it["home"], "away": it["away"],
            "market": it["market"], "prediksi": it["prediction"],
            "confidence": it["confidence"], "odds": it["odds"],
            "value": it["value"], "skor": it["skor"],
            "lapis": it["tier"], "stake_rekomendasi": it["stake"],
            "keputusan_saya": "", "stake_saya": "", "hasil": "", "profit": "", "catatan": "",
        })
    df = pd.DataFrame(rows)
    out = path("data/processed/tracking_pending.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def main():
    print("=== PREDIKSI HARI INI (dengan bobot liga) ===")
    models = load_models()
    print("[OK] 3 model loaded")
    # Validasi fixtures dulu
    is_valid, errors = validate_fixtures()
    if not is_valid:
        print("[ERROR] Fixtures tidak valid:")
        for e in errors:
            print(f"  - {e}")
        return

    fixtures = load_fixtures()
    print(f"[OK] {len(fixtures)} fixtures")
    hist = load_historical_stats()
    print(f"[OK] Historical stats: {len(hist)} baris")
    df = build_features(fixtures, hist)
    results = predict_all(models, df)
    print(f"[OK] Prediksi selesai: {len(results)} laga")
    tier_s, tier_a, tier_b = filter_and_classify(results)
    print(f"[OK] Tier S: {len(tier_s)} | A: {len(tier_a)} | B: {len(tier_b)}")

    out_json = path("data/processed/predictions_today.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_scan": len(results),
        "tier_s": tier_s, "tier_a": tier_a, "tier_b": tier_b,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[OK] JSON: {out_json}")

    txt = render_txt(tier_s, tier_a, tier_b, len(results))
    out_txt = path("data/processed/predictions_today.txt")
    out_txt.write_text(txt, encoding="utf-8")
    print(f"[OK] TXT: {out_txt}")

    save_tracking(tier_s, tier_a, tier_b)
    print(f"[OK] Tracking: data/processed/tracking_pending.csv")
    print()
    print(txt)


if __name__ == "__main__":
    main()

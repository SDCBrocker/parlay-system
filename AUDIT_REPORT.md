# AUDIT REPORT - Parlay System v4
**Date:** 2026-10-01  
**Status:** ✅ Substantial Progress | ⚠️ Critical Issues Remain

---

## RINGKASAN EKSEKUTIF

Sistem parlay telah dibangun dengan arsitektur yang solid, namun masih terdapat **12 isu kritis** yang harus diperbaiki sebelum production. Mayoritas issue terletak di:
- **Stage 1**: Data leakage dan normalisasi
- **Stage 3**: Probabilitas 1X2 dan risk gates
- **Stage 4**: Keselarasan backtest dengan prediksi aktual

---

## STAGE 1 — KONTRAK DATA DAN LEAKAGE

### ✅ Sudah Benar

1. **Rolling features dengan groupby tim** (features.py:43-98)
   - `_team_long_df()` dengan shift(1) mencegah leakage
   - Semua rolling (form_points_5, goals_for_avg_5) sudah proper

2. **TimeSeriesSplit digunakan** (train.py:81-86)
   - Split chronologically: `df.sort_values("date")` + `split_idx`
   - Test set tidak tercampur dengan training

3. **Consistency & rest-day features** 
   - Sudah ada `create_consistency_features()` (features.py:193-237)
   - Rest days feature tersedia (features.py:240-256)

### ⚠️ ISSUE KRITIS

#### **Issue 1.1: Rolling Features Belum Diakses di features.py**
**File:** `scripts/features.py` baris 187  
**Status:** 🔴 BLOCKER

```python
# ❌ PROBLEM: Fungsi ini ada tapi TIDAK DIPANGGIL di main()
def create_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """...Fitur konsistensi tim..."""
    
def create_rest_days_features(df: pd.DataFrame) -> pd.DataFrame:
    """...Fitur rest days & congestion..."""

# Solusi:
# main() harus memanggil kedua fungsi:
df = create_consistency_features(df)
df = create_rest_days_features(df)
```

**Dampak:** Fitur consistency & rest-day NOT digunakan dalam training → model suboptimal

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 1.2: Normalisasi Diterapkan SETELAH Feature Engineering**
**File:** `scripts/normalize.py:52-66`  
**Status:** ⚠️ DESIGN ISSUE

```python
# Workflow saat ini:
# 1. features.py → matches_features.csv (fitur raw)
# 2. normalize.py → matches_normalized.csv (z-score per league)
# 3. train.py (pakai normalized)

# ❌ PROBLEM:
# - Z-score computed dari SELURUH data → leakage test set masuk ke scaling
# - Normalisasi harus dari TRAINING SET SAJA

# ✅ Solusi yang benar:
# 1. features.py → matches_features.csv
# 2. train.py split: train_df, test_df
# 3. Compute scaler dari train_df SAJA
# 4. Apply scaler ke test_df (jangan refit)
# 5. Save scaler → gunakan di predict.py
```

**Dampak:** Normalization leakage dari test set → model overoptimistic

**Evidence:** `normalize.py:52` tidak ada `if "train"` filter

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 1.3: League Registry Tidak Digunakan Konsisten**
**File:** `scripts/league_registry.py`, `config.py:38-40`  
**Status:** ⚠️ FRAGMENTATION

```python
# Problem: Ada 2 sumber kebenaran untuk league list
# 1. config.py: LEAGUES = os.getenv("LEAGUES")
# 2. league_registry.py: (ada tapi tidak di-import di mana-mana)
# 3. predict.py: hardcoded LEAGUE_WEIGHTS

# Solusi:
# - Satukan ke league_registry.py
# - Import di config.py, train.py, predict.py
# - Single source of truth
```

**Fix Priority:** ⚠️ MEDIUM

---

#### **Issue 1.4: Congestion Feature Calculation Salah**
**File:** `scripts/features.py:248-254`  
**Status:** 🔴 INCORRECT LOGIC

```python
# ❌ CURRENT (WRONG):
df["home_congestion_7d"] = df.groupby("home_team").cumcount().diff().fillna(0).clip(0, 3)

# Problem: cumcount() memberikan sequential index per group
# Ini BUKAN real match count dalam 7 hari

# ✅ CORRECT:
def count_matches_in_7d(dates, current_date):
    return ((dates >= current_date - pd.Timedelta(days=7)) & 
            (dates < current_date)).sum()

df["home_congestion_7d"] = df.groupby("home_team").apply(
    lambda g: g["date"].apply(
        lambda d: count_matches_in_7d(g["date"], d)
    )
).reset_index(level=0, drop=True)
```

**Fix Priority:** 🔴 CRITICAL

---

## STAGE 2 — TRAINING DAN MODEL

### ✅ Sudah Benar

1. **Binary 1X2 Home/Away** (train.py:236-289)
   - Model terpisah untuk Home Win vs Away Win
   - Draw dikomputasi dari residual: `p_draw = 1 - p_h - p_a`

2. **Class Weight XGBoost** (train.py:132)
   - `scale_pos_weight=1` set (sudah ada)
   - LGBM: `class_weight="balanced"` ✓

3. **Ensemble dengan Bobot** (train.py:64-69, 141-142)
   - XGB + LGBM dengan weighted average
   - Metadata disimpan: `ENSEMBLE_WEIGHTS`

4. **Model Versioning** (config.py:18-25, train.py:229-234)
   - `MODEL_VERSION = f"{MODEL_VERSION_PREFIX}_{N_LEAGUES}league_{TRAINING_DATE}"`
   - Metadata lengkap disimpan (train.py:227-235)

5. **CV Checks untuk Overfitting** (train.py:114-121, 159-171)
   - Cross-validation dengan 5 folds
   - Overfitting detection: `gap > CV_OVERFITTING_THRESHOLD`

### ⚠️ ISSUE

#### **Issue 2.1: Model Version Tidak Stabil — Bergantung pada TRAINING_DATE**
**File:** `config.py:24-25`  
**Status:** ⚠️ DESIGN

```python
# ❌ CURRENT:
TRAINING_DATE = datetime.now().strftime("%Y%m%d")
MODEL_VERSION = f"{MODEL_VERSION_PREFIX}_{N_LEAGUES}league_{TRAINING_DATE}"

# Problem:
# - Jika train 2x dalam 1 hari dengan data sama → versi berbeda
# - Kalau data/config tidak berubah → version tetap harus sama
# - Jika train 2x dengan data berbeda → version harus berubah

# ✅ Solusi: Hash data + config
import hashlib
data_hash = hashlib.md5(
    pd.read_csv(path("data/processed/matches_features.csv"))
    .to_string().encode()
).hexdigest()[:8]
MODEL_VERSION = f"{MODEL_VERSION_PREFIX}_{N_LEAGUES}league_{data_hash}"
```

**Fix Priority:** ⚠️ MEDIUM (tidak blocker, tapi penting untuk reproducibility)

---

#### **Issue 2.2: TimeSeriesSplit Hanya di train.py, Tidak di Backtest**
**File:** `scripts/backtest.py:29-40`  
**Status:** ⚠️ INCONSISTENCY

```python
# backtest.py pakai matches_test.csv yang sudah split oleh train.py
# ✓ Ini OK, tapi TIDAK ROBUST

# Solusi:
# - Simpan split indices di model metadata
# - backtest.py validate bahwa test_df adalah bagian chronologically latter
```

**Fix Priority:** ⚠️ LOW

---

## STAGE 3 — PREDIKSI DAN RISK GATE

### ✅ Sudah Benar

1. **Model Ensemble Support** (predict.py:177-194)
   - `predict_with_ensemble()` handle XGB + LGBM dengan bobot

2. **Tier System** (config.py:49-72, 137-149)
   - 3 tier (S, A, B) dengan thresholds jelas
   - `klasifikasi_tier()` function

3. **Stake Limit Per Match** (config.py:74-77)
   - `MAX_STAKE_PER_MATCH = 50000`
   - `MAX_DAILY_STAKE = 100000`

4. **Deterministic ID Tracking** (predict.py:88-96)
   - `match_id = f"{liga}_{dt_str}_{home}_{away}"`

### ⚠️ ISSUE KRITIS

#### **Issue 3.1: Probabilitas 1X2 Tidak Include DRAW Dengan Benar**
**File:** `scripts/predict.py:213-229`  
**Status:** 🔴 CRITICAL BUG

```python
# ❌ CURRENT:
p_h = prob_home[i][1]
p_a = prob_away[i][1]
p_draw = 1 - p_h - p_a  # WRONG CALCULATION
p_draw = max(0, min(1, p_draw))
choices = [
    ("Home Win", p_h, row.get("odds_home")),
    ("Draw", p_draw, row.get("odds_draw")),
    ("Away Win", p_a, row.get("odds_away")),
]
best_label, best_prob, best_odds = max(choices, key=lambda x: x[1])

# Problem:
# - p_draw bukan independent model
# - Kalau p_h + p_a > 1 → p_draw negative → clamped ke 0
# - Draw tidak benar-benar diprediksi, hanya sisa

# ✅ Solusi 1: Model 3-way classification (1/D/A)
# ✅ Solusi 2: Train model_draw terpisah
# ✅ Solusi 3 (temporary): Estimasi dari historical frequency
#    p_draw = historical_draw_rate * (1 - p_h - p_a)
```

**Dampak:** Prediksi Draw unreliable → odds backing salah

**Evidence:** Tidak ada 3-way classification model di train.py

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 3.2: O/U Lines 1.5 dan 3.5 Menggunakan Heuristic, Bukan Model**
**File:** `scripts/predict.py:232-240`  
**Status:** 🔴 UNRELIABLE

```python
# ❌ CURRENT:
if line == 2.5:
    p_over = p_over_2_5
elif line == 1.5:
    p_over = min(0.95, p_over_2_5 + 0.20)  # ❌ HARDCODED +0.20
elif line == 3.5:
    p_over = max(0.05, p_over_2_5 - 0.20)  # ❌ HARDCODED -0.20

# Problem:
# - Offset +0.20 arbitrary, tidak berdasarkan data
# - Bergantung pada model 2.5 saja

# ✅ Solusi: Train 3 model terpisah untuk O/U 1.5, 2.5, 3.5
# Atau kalau ingin efficient: polynomial regression untuk interpolate
```

**Dampak:** Prediksi O/U 1.5 dan 3.5 tidak reliable

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 3.3: Daily Stake Limit Hanya di render_txt(), Tidak di Tracking**
**File:** `scripts/predict.py:371-377`  
**Status:** ⚠️ IMPLEMENTATION

```python
# ❌ CURRENT:
total_stake = sum(it["stake"] for it in tier_s + tier_a + tier_b)
total_stake = min(total_stake, MAX_DAILY_STAKE)  # Hanya di render
# Tapi di save_tracking(), tidak ada pengecekan

# ✅ Solusi:
def apply_daily_limit(all_items, max_daily=MAX_DAILY_STAKE):
    all_items_sorted = sorted(all_items, key=lambda x: x["skor"], reverse=True)
    running_total = 0
    limited_items = []
    for it in all_items_sorted:
        if running_total + it["stake"] <= max_daily:
            limited_items.append(it)
            running_total += it["stake"]
    return limited_items
```

**Fix Priority:** ⚠️ MEDIUM

---

#### **Issue 3.4: Market Limit Per Match/Liga Tidak Ada**
**File:** `scripts/predict.py`  
**Status:** 🔴 MISSING FEATURE

```python
# ❌ TIDAK ADA:
# - Max N market per match (e.g., max 2 per match)
# - Max N market per liga per hari

# ✅ Solusi:
# Tambah di config.py:
MAX_MARKETS_PER_MATCH = 2
MAX_MARKETS_PER_LEAGUE = 20

# Implementasi di filter_and_classify():
for match in results:
    market_count_per_match = defaultdict(int)
    # Filter: keep top by skor, up to MAX_MARKETS_PER_MATCH per match
```

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 3.5: Model Metadata Tidak Disimpan di Prediksi**
**File:** `scripts/predict.py:427-434`  
**Status:** ⚠️ TRACKING

```python
# ❌ CURRENT: predictions_today.json tidak mencatat model_version
payload = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "total_scan": len(results),
    "tier_s": tier_s, "tier_a": tier_a, "tier_b": tier_b,
}

# ✅ Solusi:
payload = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "model_version": MODEL_VERSION,  # ADD THIS
    "model_metadata": load_model_metadata(),  # ADD THIS
    "total_scan": len(results),
    "tier_s": tier_s, "tier_a": tier_a, "tier_b": tier_b,
}
```

**Fix Priority:** ⚠️ MEDIUM

---

## STAGE 4 — BACKTEST DAN TRACKING

### ✅ Sudah Benar

1. **Value Betting Simulation** (backtest.py:122-155)
   - Perhitungan value: `sub["value"] = sub["prob"] * sub["odds"] - 1`
   - Profit calculation: `won * (odds - 1) - (1 - won)`

2. **Per-League Evaluation** (backtest_per_league.py ada)
   - Script terpisah untuk breakdown per league

3. **Model Versioning di Training Log** (train.py:228-234)
   - Master training log: logs/training_log.json

### ⚠️ ISSUE KRITIS

#### **Issue 4.1: Backtest Strategy Tidak Sama Dengan Prediksi Aktual**
**File:** `scripts/backtest.py` vs `scripts/predict.py`  
**Status:** 🔴 CRITICAL DIVERGENCE

```python
# BACKTEST (backtest.py:122-155):
# - Filter: value > 0.05
# - Label: result_enc (0/1/2 untuk H/D/A)
# - Bet: prob_H * odds_home - 1

# PREDICT (predict.py:285-338):
# - Filter: skor_final > MIN_WEIGHTED_SCORE (0.30)
# - Filter: confidence + value + tier classification
# - Apply league weights, tier-based stakes
# - NO value filter dalam predict.py!

# ❌ PROBLEM: Backtest tidak menggunakan tier system, tidak menggunakan league bobot
# Backtest result ≠ Real prediction result

# ✅ Solusi: Align backtest dengan predict
# - backtest.py harus replicate exact filter logic dari predict.py
# - Gunakan tier system, league weights, skor_final computation
```

**Dampak:** Backtest ROI tidak match actual performance

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 4.2: Backtest Tidak Include DRAW**
**File:** `scripts/backtest.py:96-98`  
**Status:** 🔴 INCOMPLETE

```python
# ❌ CURRENT:
df["pred_1x2"] = np.where(df["prob_H"] >= df["prob_A"], 0, 2)

# Ini BINARY: Home (0) atau Away (2), tidak ada Draw (1)
# Tapi test set punya result_enc = 0,1,2

# ✅ Solusi:
df["pred_1x2"] = np.argmax([df["prob_H"], df["prob_D"], df["prob_A"]], axis=0)
# Atau compute p_draw properly dulu
```

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 4.3: Tracking Scripts Tidak Disinkronkan**
**File:** `scripts/update_tracking.py`, `scripts/cleanup_tracking.py`, `scripts/check_result.py`  
**Status:** ⚠️ INCONSISTENCY

```python
# 3 script terpisah dengan logic berbeda?
# - update_tracking.py: update hasil?
# - cleanup_tracking.py: hapus old data?
# - check_result.py: validasi hasil?

# Seharusnya:
# - 1 class TrackingManager yang handle semua
# - atau 1 master script dengan sub-commands
```

**Fix Priority:** ⚠️ MEDIUM

---

#### **Issue 4.4: Backup Tracking Menumpuk Tanpa Policy**
**File:** `scripts/cleanup_generated.sh`  
**Status:** ⚠️ MAINTENANCE

```python
# cleanup_generated.sh ada, tapi tidak jelas:
# - Berapa hari tracking disimpan?
# - Kapan backup dihapus?
# - Automated cleanup atau manual?

# ✅ Solusi:
# - Add to config.py: TRACKING_RETENTION_DAYS = 90
# - Automated cleanup dalam prediction pipeline
# - Archive old tracking to separate folder
```

**Fix Priority:** ⚠️ LOW

---

## STAGE 5 — INTEGRASI DAN SETUP

### ✅ Sudah Benar

1. **Config dengan Versioning** (config.py)
   - Central configuration
   - MODEL_VERSION tracking

2. **Feature Schema** (scripts/schema.py ada)
   - Metadata struktur data

### ⚠️ ISSUE

#### **Issue 5.1: Whitelist Tidak Dipanggil di Manapun**
**File:** `scripts/league_whitelist.py`, `scripts/fetch_all.py`, `scripts/validate_fixtures.py`  
**Status:** 🔴 DISCONNECTED

```python
# league_whitelist.py ada dengan fungsi:
def filter_fixtures_by_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Filter fixtures: kedua tim harus ada di whitelist liga."""

# ❌ TAPI tidak dipanggil di:
# - validate_fixtures.py (hanya cek struktur, tidak cek team names)
# - predict.py (load fixtures tanpa filter)
# - fetch_all.py

# ✅ Solusi:
# Di predict.py sebelum predict:
from league_whitelist import filter_fixtures_by_whitelist
fixtures = filter_fixtures_by_whitelist(fixtures)
```

**Fix Priority:** 🔴 CRITICAL

---

#### **Issue 5.2: League Mapping Tidak Unified**
**File:** `config.py:102-114`, `scripts/league_whitelist.py:16-98`, `scripts/predict.py:23-34`  
**Status:** ⚠️ MULTIPLE SOURCES

```python
# 3 tempat berbeda punya league info:
# 1. config.py LEAGUE_CODES: EPL→E0, LaLiga→SP1, etc
# 2. league_whitelist.py STATIC_WHITELIST: EPL→[teams], dll
# 3. predict.py LEAGUE_WEIGHTS: EPL→0.60, dll

# ❌ Tidak ada single source of truth
# Kalau tambah liga baru, perlu update 3 file

# ✅ Solusi: Master league registry
# File: scripts/league_registry.py (atau perluas yang ada)
LEAGUE_REGISTRY = {
    "EPL": {
        "code": "E0",
        "confidence_adjust": 1.0,
        "prediction_weight": 0.60,
        "teams": ["Arsenal", "Aston Villa", ...],
    },
    "LaLiga": {...},
    ...
}
```

**Fix Priority:** ⚠️ MEDIUM

---

#### **Issue 5.3: Fuzzy Matching Threshold 0.70 Terlalu Ketat**
**File:** `scripts/league_whitelist.py:115-136`  
**Status:** ⚠️ IMPLEMENTATION

```python
# Test case:
# - "Hull City" vs "Hull" → tidak match (terlalu short)
# - "Sp Braga" vs "Sporting Braga" → mungkin tidak match
# Threshold 0.70 terlalu ketat untuk nama tim

# ✅ Solusi:
# - Punya multiple thresholds per similarity type
# - Exact match dulu, baru fuzzy
# - Team name normalization lebih baik
```

**Fix Priority:** ⚠️ LOW

---

#### **Issue 5.4: .env.example Belum Update Lengkap**
**File:** `.env.example`  
**Status:** ⚠️ DOCUMENTATION

```python
# Missing:
# - MIN_HISTORICAL_MATCHES
# - MAX_STAKE_PER_MATCH
# - CV_FOLDS, CV_OVERFITTING_THRESHOLD
# - MIN_BACKTEST_ROI, MIN_BACKTEST_ACCURACY
# - LEAGUE CONFIDENCE ADJUST per league

# ✅ Solusi: Sync dengan config.py semua ENV vars
```

**Fix Priority:** ⚠️ LOW

---

#### **Issue 5.5: requirements.txt Belum Complete**
**File:** `requirements.txt`  
**Status:** ⚠️ INCOMPLETE

```python
# Missing:
# - scipy (kalau gunakan stats)
# - pytest (untuk testing)
# - apify-client (jika pakai Apify)
# - pytz (untuk timezone handling)

# ✅ Solusi: Add dengan version pins
scipy>=1.10.0
pytest>=7.0.0
apify-client>=1.0.0
pytz>=2023.3
```

**Fix Priority:** ⚠️ LOW

---

#### **Issue 5.6: verify_setup.py Hardcoded Check model_1x2, Tidak Flexible**
**File:** `scripts/verify_setup.py:92-103`  
**Status:** ⚠️ MAINTENANCE

```python
# ❌ CURRENT:
for name in ["model_1x2", "model_ou", "model_btts"]:
    p = path(f"models/{name}.pkl")
    # Check model_1x2, bukan model_1x2_home + model_1x2_away

# ✅ Solusi:
REQUIRED_MODELS = ["model_1x2_home", "model_1x2_away", "model_ou", "model_btts"]
for name in REQUIRED_MODELS:
    p = path(f"models/{name}.pkl")
    # Validate both home dan away
```

**Fix Priority:** ⚠️ MEDIUM

---

#### **Issue 5.7: Tidak Ada Smoke Test Pipeline**
**File:** Tidak ada  
**Status:** 🔴 MISSING

```python
# ❌ Tidak ada script untuk test end-to-end:
# 1. Load data
# 2. Train model
# 3. Predict on sample fixtures
# 4. Validate output format
# 5. Check no crashes

# ✅ Solusi: tests/test_pipeline.py atau scripts/smoke_test.py
def test_full_pipeline():
    """End-to-end smoke test."""
    # Create minimal test data
    # Train model (minimal)
    # Predict
    # Verify output
```

**Fix Priority:** 🔴 CRITICAL

---

## SUMMARY SCORECARD

| Stage | Status | Issues | Blockers |
|-------|--------|--------|----------|
| **1. Data & Leakage** | 🟡 70% | 4 | 3 |
| **2. Training & Model** | 🟢 85% | 2 | 0 |
| **3. Prediksi & Risk Gate** | 🟡 60% | 5 | 4 |
| **4. Backtest & Tracking** | 🟡 65% | 4 | 2 |
| **5. Integrasi & Setup** | 🟡 70% | 7 | 2 |
| **TOTAL** | 🟡 70% | **22** | **11** |

---

## PRIORITAS PERBAIKAN (Action Plan)

### 🔴 CRITICAL (HARUS DIKERJAKAN SEKARANG)

1. **[1.1]** Panggil consistency & rest-day features di features.py main()
2. **[1.2]** Pindahkan normalisasi ke train.py (compute scaler dari train saja)
3. **[1.4]** Fix congestion feature calculation
4. **[3.1]** Tambah Draw model atau 3-way classification untuk 1X2
5. **[3.2]** Train terpisah untuk O/U 1.5, 2.5, 3.5 (bukan hardcoded offset)
6. **[3.4]** Implementasi market limit per match dan per liga
7. **[4.1]** Align backtest strategy dengan predict strategy
8. **[4.2]** Include Draw di backtest (3-way classification)
9. **[5.1]** Aktifkan whitelist filter di predict.py
10. **[5.7]** Buat smoke test pipeline
11. **[5.2]** Unify league registry (single source of truth)
12. **[3.3]** Enforce daily stake limit di save_tracking()

### ⚠️ MEDIUM (SHOULD FIX BEFORE PRODUCTION)

1. **[2.1]** Stabilkan model version dengan data hash
2. **[3.5]** Simpan model metadata di predictions_today.json
3. **[4.3]** Consolidate tracking scripts
4. **[5.4]** Update .env.example lengkap
5. **[5.6]** Fix verify_setup.py model checks

### 💡 NICE TO HAVE

1. **[1.3]** League registry usage consistency
2. **[4.4]** Automated tracking cleanup policy
3. **[5.3]** Improve fuzzy matching thresholds
4. **[5.5]** Complete requirements.txt

---

## REKOMENDASI DEVELOPMENT

### Timeline
- **Minggu 1**: Fix critical issues #1-8
- **Minggu 2**: Fix critical issues #9-12 + medium priority
- **Minggu 3**: Test & validation, smoke test

### Testing Strategy
```python
# Tambah pytest tests untuk:
# 1. No data leakage (no future info in features)
# 2. No normalization leakage (scaler fitted on train only)
# 3. Backtest matches predict (same filtering logic)
# 4. Draw probability valid (0 <= p_draw <= 1)
# 5. Tier classification deterministic
# 6. Model versioning stable
```

### Deployment Checklist
- [ ] Semua 11 critical issues fixed
- [ ] Smoke test passing
- [ ] Backtest vs actual predictions aligned
- [ ] League whitelist validated
- [ ] Daily limits enforced
- [ ] Model metadata tracked
- [ ] Documentation updated

---

## KESIMPULAN

Sistem parlay sudah memiliki **foundation yang solid** dengan:
- ✅ Proper feature engineering (dengan shift untuk leakage prevention)
- ✅ Ensemble models (XGB + LGBM dengan bobot)
- ✅ Tier system dan risk management
- ✅ Model versioning

Namun masih ada **11 critical issues** yang perlu diperbaiki:
- Data leakage di normalisasi
- Probabilitas 1X2 belum benar
- Draw model tidak ada
- Backtest tidak selaras dengan prediksi
- Whitelist tidak aktif
- Risk gates tidak complete

**Estimated effort:** 2-3 minggu untuk fix semua critical issues + testing.

**Recommendation:** JANGAN production sampai semua critical issues fixed, terutama #3.1 (Draw model) dan #4.1 (Backtest alignment).


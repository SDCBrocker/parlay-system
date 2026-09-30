"""
utils.py — Helper umum untuk semua script.
"""
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import path, CACHE_DIR


# === Cache ===
def cache_key(endpoint: str, params: dict) -> str:
    raw = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_path(endpoint: str, params: dict) -> Path:
    return CACHE_DIR / f"{cache_key(endpoint, params)}.json"


def load_cache(endpoint: str, params: dict, ttl_hours: int = 24):
    p = cache_path(endpoint, params)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > ttl_hours * 3600:
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        p.unlink()  # cache korup → hapus
        return None


def save_cache(endpoint: str, params: dict, data):
    p = cache_path(endpoint, params)
    p.write_text(json.dumps(data, indent=2))


# === CSV ===
def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def save_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


# === Backup ===
def backup_file(p: Path) -> Path:
    if not p.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path("data/backup") / f"{p.stem}_{ts}{p.suffix}"
    shutil.copy2(p, dest)
    return dest


# === Deduplikasi ===
def deduplicate(df: pd.DataFrame, key: str = "match_id") -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return df
    return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


# === Merge ===
def merge_database(df_main: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Gabung df_main + df_new dengan skema sama.
    df_new menang kalau match_id sama.
    """
    if df_main.empty:
        return df_new.copy()
    if df_new.empty:
        return df_main.copy()
    combined = pd.concat([df_main, df_new], ignore_index=True)
    combined = deduplicate(combined)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def rotate_log(log_path, max_size_mb: float = 5.0):
    """
    Rotate log kalau > max_size_mb.
    Rename ke .log.1, hapus .log.1 lama.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return

    size_mb = log_path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return

    # Rename .log.1 lama (hapus)
    rotated = log_path.with_suffix(log_path.suffix + ".1")
    if rotated.exists():
        rotated.unlink()

    # Rename log lama ke .log.1
    log_path.rename(rotated)
    print(f"[LOG ROTATE] {log_path.name} ({size_mb:.1f} MB) -> {rotated.name}")

"""
api_rotator.py — Rotasi API key otomatis.
Kalau key1 kena limit (429), pindah ke key2, dst.
"""
import os
from itertools import cycle


class APIKeyRotator:
    def __init__(self, keys_str: str, name: str = "API"):
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys:
            raise ValueError(f"{name}_KEYS kosong di .env")
        self.keys = keys
        self.name = name
        self.cycle = cycle(keys)
        self.current = next(self.cycle)
        self.exhausted = set()
        print(f"[{name}] Loaded {len(keys)} key(s)")

    def get(self) -> str:
        """Ambil key berikutnya yang belum exhausted."""
        if len(self.exhausted) >= len(self.keys):
            raise RuntimeError(f"[{self.name}] Semua key exhausted")
        # Cari key yang belum exhausted
        for _ in range(len(self.keys)):
            k = self.current
            self.current = next(self.cycle)
            if k not in self.exhausted:
                return k
        raise RuntimeError(f"[{self.name}] Semua key exhausted")

    def mark_exhausted(self, key: str):
        """Tandai key sebagai exhausted (kena limit)."""
        self.exhausted.add(key)
        remaining = len(self.keys) - len(self.exhausted)
        print(f"[{self.name}] Key exhausted. Sisa: {remaining}")

    def rotate(self) -> str:
        """Paksa rotate ke key berikutnya."""
        self.current = next(self.cycle)
        return self.current

    def has_available(self) -> bool:
        return len(self.exhausted) < len(self.keys)


if __name__ == "__main__":
    # Test
    rot = APIKeyRotator("k1,k2,k3", "TEST")
    print("Key 1:", rot.get())
    print("Key 2:", rot.get())
    rot.mark_exhausted(rot.current)
    print("Key 3:", rot.get())
    print("Available:", rot.has_available())

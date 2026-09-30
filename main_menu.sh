#!/data/data/com.termux/files/usr/bin/bash
# main_menu.sh — Entry point menu utama

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Cek venv
if [ ! -d "venv" ]; then
    echo "venv tidak ditemukan. Jalankan dulu: bash setup.sh"
    exit 1
fi

# Aktifkan venv
source venv/bin/activate

# Jalankan menu
if [ ! -f "menus/main_menu.py" ]; then
    echo "menus/main_menu.py belum ada (akan dibuat di TAHAP 6)."
    echo "Sementara, test config dulu:"
    python config.py
    exit 0
fi

python menus/main_menu.py

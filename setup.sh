#!/data/data/com.termux/files/usr/bin/bash
# setup.sh — Setup otomatis sistem prediksi parlay di Termux
set -e

# Warna
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Setup Sistem Prediksi Parlay ===${NC}"

# 1. Cek Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 tidak ditemukan. Install dulu: pkg install python${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 ditemukan: $(python3 --version)${NC}"

# 2. Cek git
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠ git tidak ditemukan. Install dulu: pkg install git${NC}"
    exit 1
fi
echo -e "${GREEN}✓ git ditemukan${NC}"

# 3. Tentukan root project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
echo -e "${GREEN}✓ Root project: $ROOT_DIR${NC}"

# 4. Buat venv kalau belum ada
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Membuat virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ venv dibuat${NC}"
else
    echo -e "${GREEN}✓ venv sudah ada${NC}"
fi

# 5. Aktifkan venv & upgrade pip
source venv/bin/activate
pip install --upgrade pip --quiet

# 6. Install dependencies
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Menginstall dependencies...${NC}"
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies terinstall${NC}"
else
    echo -e "${RED}requirements.txt tidak ditemukan${NC}"
    exit 1
fi

# 7. Buat struktur folder
echo -e "${YELLOW}Membuat struktur folder...${NC}"
mkdir -p data/raw/cache
mkdir -p data/processed
mkdir -p data/backup
mkdir -p models
mkdir -p scripts
mkdir -p menus
mkdir -p logs
mkdir -p notebooks
echo -e "${GREEN}✓ Struktur folder dibuat${NC}"

# 8. Copy .env.example ke .env kalau belum ada
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ .env dibuat dari .env.example${NC}"
        echo -e "${YELLOW}⚠ Jangan lupa isi API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID di .env${NC}"
    else
        echo -e "${RED}.env.example tidak ditemukan${NC}"
    fi
else
    echo -e "${GREEN}✓ .env sudah ada${NC}"
fi

# 9. Ringkasan
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}   SETUP SELESAI${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "Root      : $ROOT_DIR"
echo -e "Venv      : $ROOT_DIR/venv"
echo -e "Python    : $(python3 --version)"
echo ""
echo -e "${YELLOW}Langkah selanjutnya:${NC}"
echo -e "  1. nano .env          → isi API_KEY, TELEGRAM_TOKEN, CHAT_ID"
echo -e "  2. bash main_menu.sh  → jalankan menu utama"
echo -e "${GREEN}════════════════════════════════════════${NC}"

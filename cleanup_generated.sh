#!/bin/bash
# cleanup_generated.sh
# Safe cleanup for generated data only.
# Does NOT delete scripts/, menus/, config.py, or README.

set -e

cd "$(dirname "$0")"

echo "=== CLEANUP GENERATED FILES ONLY ==="

# Remove tracked generated files by extension/folder
for pattern in \
  "data/*" \
  "data/raw/*" \
  "data/processed/*" \
  "data/backup/*" \
  "models/*" \
  "models/backup/*" \
  "logs/*" \
  "notebooks/*" \
  "*.bak" \
  "*.csv" \
  "*.log" \
  "*.tmp"; do
  find . -path "./.git" -prune -o -type f \( -name "*.bak" -o -name "*.csv" -o -name "*.log" -o -name "*.tmp" \) -print -delete 2>/dev/null || true
  find . -path "./.git" -prune -o -type f -path "./data/*" -print -delete 2>/dev/null || true
  find . -path "./.git" -prune -o -type f -path "./models/*" -print -delete 2>/dev/null || true
  find . -path "./.git" -prune -o -type f -path "./logs/*" -print -delete 2>/dev/null || true
  find . -path "./.git" -prune -o -type f -path "./notebooks/*" -print -delete 2>/dev/null || true
  break
done

# Recreate empty directories so repo stays structured
mkdir -p data/raw data/processed data/backup models/backup logs notebooks

# Keep folder structure with markers only
for f in \
  data/.gitkeep \
  data/raw/.gitkeep \
  data/processed/.gitkeep \
  data/backup/.gitkeep \
  models/.gitkeep \
  models/backup/.gitkeep \
  logs/.gitkeep \
  notebooks/.gitkeep; do
  : > "$f"
done

# Remove generated files from git tracking if they are tracked
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files | grep -E '(^data/|^models/|^logs/|^notebooks/|\.bak$|\.csv$|\.log$|\.tmp$)' | xargs -r git rm -f --ignore-unmatch || true
  git add data/.gitkeep data/raw/.gitkeep data/processed/.gitkeep data/backup/.gitkeep \
    models/.gitkeep models/backup/.gitkeep logs/.gitkeep notebooks/.gitkeep
  echo "Git tracking cleaned for generated files."
fi

echo "Cleanup done. Scripts and source files are preserved."

#!/usr/bin/env bash
# ==============================================================================
# sync_wiki.sh
# Safely synchronize documentation from local wiki/ directory to GitHub Wiki.
#
# Usage (run from repository root in WSL):
#   bash scripts/sync_wiki.sh
# ==============================================================================

set -euo pipefail

WIKI_REMOTE="https://github.com/rlica/python-cmat.wiki.git"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_WIKI_DIR="${REPO_ROOT}/wiki"
TEMP_DIR="/tmp/python-cmat-wiki-$$"

echo "=== python-cmat GitHub Wiki Synchronization ==="
echo "Local source: ${LOCAL_WIKI_DIR}"
echo "Wiki remote:  ${WIKI_REMOTE}"
echo

if [ ! -d "${LOCAL_WIKI_DIR}" ]; then
  echo "Error: Local wiki directory not found at ${LOCAL_WIKI_DIR}" >&2
  exit 1
fi

# Clean up temporary directory on exit
trap 'rm -rf "${TEMP_DIR}"' EXIT

echo "[*] Cloning remote wiki repository..."
git clone "${WIKI_REMOTE}" "${TEMP_DIR}"

echo "[*] Copying wiki pages..."
cp -v "${LOCAL_WIKI_DIR}"/*.md "${TEMP_DIR}/"

cd "${TEMP_DIR}"

# Inherit git credentials/ident from parent repo if needed
GIT_USER_NAME="$(git -C "${REPO_ROOT}" config user.name || echo 'rlica')"
GIT_USER_EMAIL="$(git -C "${REPO_ROOT}" config user.email || echo 'razvanlica@rlicas-MacBook-Air-54.local')"
git config user.name "${GIT_USER_NAME}"
git config user.email "${GIT_USER_EMAIL}"

git add -A

if git diff --cached --quiet; then
  echo
  echo "[✓] Wiki is already up to date. No changes to push."
  exit 0
fi

echo
echo "=== Staged Wiki Changes ==="
git status -s
echo

read -rp "Are you sure you want to commit and push these changes to GitHub Wiki? [y/N]: " CONFIRM
if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
  git commit -m "Update documentation from local wiki/ $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "[*] Pushing changes to GitHub Wiki..."
  git push origin HEAD
  echo "[✓] GitHub Wiki successfully updated! Visit: https://github.com/rlica/python-cmat/wiki"
else
  echo "[!] Push cancelled by user. No changes were sent to GitHub."
  exit 0
fi

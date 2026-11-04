#!/usr/bin/env bash
# Kiem nhanh truoc khi chay that: pymol2, audit split, cac test an toan. Dung o dau khi that bai.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 12.0  pymol2 =="
python -c "import pymol2; p=pymol2.PyMOL(); p.start(); print('pymol', p.cmd.get_version()); p.stop()"

echo "== 12.5  audit split =="
python scripts/00_audit_split.py

echo "== tests =="
python -m pytest tests -q

echo "== OK =="

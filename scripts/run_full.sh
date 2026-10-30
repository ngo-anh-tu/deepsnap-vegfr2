#!/usr/bin/env bash
# Chay toan bo pipeline V1 tren may remote: check -> conformer -> render -> baseline -> train.
# KHONG chay test set (K.7): buoc do la 08_final_test.py, chay tay sau khi chot cau hinh.
#   nohup bash scripts/run_full.sh > logs/run_full.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

NCPU=$(python -c "import os; print(os.cpu_count())")
W=$(( NCPU < 16 ? NCPU : 16 ))
PYMOL_WRAP=""
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then PYMOL_WRAP="xvfb-run -a"; fi

step() { echo; echo "=== [$(date +%H:%M:%S)] $1"; }

step "0. kiem tra moi truong + test"
$PYMOL_WRAP bash scripts/00_check_pipeline.sh 2>&1 | tee logs/00_check.log

step "1. conformer + geometry.parquet"
python scripts/01_build_conformers.py --workers "$W" 2>&1 | tee logs/01_conformers.log

step "2. render PNG (tu bo qua anh da co; chay lai an toan)"
$PYMOL_WRAP python scripts/02_render.py --config configs/base.yaml configs/render_v1.yaml \
    --workers "$W" 2>&1 | tee logs/02_render.log

step "3. contact sheet de luot bang mat"
python scripts/09_contact_sheet.py --config configs/base.yaml configs/render_v1.yaml 2>&1 | tee logs/09_sheet.log

step "4. baseline (hang so + ECFP4/GBM tren split_all.csv)"
python baselines/constant.py 2>&1 | tee logs/04_baseline_constant.log
python baselines/ecfp_gbm.py 2>&1 | tee logs/04_baseline_ecfp.log

step "5. train V1 + ResNet-18, 3 seed (chi doc train+val)"
python scripts/03_train.py 2>&1 | tee logs/05_train.log

echo
echo "XONG. Xem: data/renders/v1/qa/ , runs/v1_r18/metrics.json , runs/baseline_*/"
echo "Chot cau hinh roi moi chay test (DUNG MOT LAN):"
echo "  python scripts/08_final_test.py runs/v1_r18 --reason \"...\""

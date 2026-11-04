#!/usr/bin/env bash
# Chay toan bo pipeline tren data/raw/split_all.csv:
#   check -> conformer -> render -> contact sheet -> baseline -> train (train+val) -> metric moi split
#   -> [tuy chon] test set DUNG MOT LAN (K.7).
#
#   nohup bash scripts/run_full.sh > logs/run_full.out 2>&1 &
#   RENDER=v1_hq2_512 MODEL=r18 nohup bash scripts/run_full.sh > logs/run_full_v1_hq2_512.out 2>&1 &
#   RENDER=v1_hq2 FINAL_TEST=1 REASON="cau hinh da chot" bash scripts/run_full.sh
#   RENDER=v9_hq SKIP_CHECK=1 SKIP_BASELINES=1 nohup bash scripts/run_full.sh > logs/run_full_v9_hq.out 2>&1 &
#
# Bien moi truong:
#   RENDER          configs/render_<RENDER>.yaml   (mac dinh v1_hq2)
#   MODEL           configs/model_<MODEL>.yaml     (mac dinh r18)
#   SKIP_CHECK=1    bo buoc 0 (pymol2 + audit split + tests)
#   SKIP_BASELINES=1  bo buoc 4 (baseline khong phu thuoc RENDER/MODEL, chi can chay mot lan)
#   FINAL_TEST=1 + REASON="..."  cham test sau khi train; 08_final_test.py tu choi lan hai tren cung run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

RENDER=${RENDER:-v1_hq2}
MODEL=${MODEL:-r18}
RCFG=configs/render_${RENDER}.yaml
MCFG=configs/model_${MODEL}.yaml
for f in "$RCFG" "$MCFG"; do [ -s "$f" ] || { echo "thieu hoac rong: $f"; exit 1; }; done
if [ "${FINAL_TEST:-0}" = 1 ] && [ -z "${REASON:-}" ]; then
    echo 'FINAL_TEST=1 can REASON="..." (ghi vao reports/test_set_log.md)'; exit 1
fi

# run_name = <render.name>_<model.name> (src/config.py), doc tu config thay vi doan tu ten file
RUN=$(python -c "
import sys; sys.path.insert(0, '.')
from src.config import load_config, run_name
print(run_name(load_config('configs/base.yaml', '$RCFG', '$MCFG')))")
RUN_DIR=runs/$RUN

NCPU=$(python -c "import os; print(os.cpu_count())")
W=$(( NCPU < 16 ? NCPU : 16 ))
PYMOL_WRAP=""
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then PYMOL_WRAP="xvfb-run -a"; fi

step() { echo; echo "=== [$(date +%H:%M:%S)] $1"; }
echo "render=$RCFG  model=$MCFG  run=$RUN_DIR  workers=$W"

if [ "${SKIP_CHECK:-0}" != 1 ]; then
    step "0. kiem tra moi truong + test"
    $PYMOL_WRAP bash scripts/00_check_pipeline.sh 2>&1 | tee logs/00_check.log
fi

step "1. conformer + geometry.parquet"
python scripts/01_build_conformers.py --workers "$W" 2>&1 | tee logs/01_conformers.log

step "2. render PNG $RENDER (tu bo qua anh da co; chay lai an toan)"
$PYMOL_WRAP python scripts/02_render.py --config configs/base.yaml "$RCFG" \
    --workers "$W" 2>&1 | tee "logs/02_render_$RENDER.log"

step "3. contact sheet de luot bang mat"
python scripts/09_contact_sheet.py --config configs/base.yaml "$RCFG" 2>&1 | tee "logs/09_sheet_$RENDER.log"

if [ "${SKIP_BASELINES:-0}" != 1 ]; then
    step "4. baseline (hang so + ECFP4/GBM tren split_all.csv)"
    python baselines/constant.py 2>&1 | tee logs/04_baseline_constant.log
    python baselines/ecfp_gbm.py 2>&1 | tee logs/04_baseline_ecfp.log
fi

step "5. train $RUN, 3 seed (chi doc train+val)"
python scripts/03_train.py --config configs/base.yaml "$RCFG" "$MCFG" 2>&1 | tee "logs/05_train_$RENDER.log"

step "6. metric train/val (+ test neu da cham) -> $RUN_DIR/metrics_all_splits.csv"
python scripts/10_split_metrics.py "$RUN_DIR" 2>&1 | tee "logs/10_metrics_$RENDER.log"

if [ "${FINAL_TEST:-0}" = 1 ]; then
    step "7. TEST SET (mot lan duy nhat): $REASON"
    python scripts/08_final_test.py "$RUN_DIR" --reason "$REASON" 2>&1 | tee "logs/08_final_test_$RENDER.log"
    python scripts/10_split_metrics.py "$RUN_DIR" 2>&1 | tee -a "logs/10_metrics_$RENDER.log"
fi

echo
echo "XONG. Xem: data/renders/$RENDER/qa/ , $RUN_DIR/metrics.json , $RUN_DIR/metrics_all_splits.csv , runs/baseline_*/"
if [ "${FINAL_TEST:-0}" != 1 ]; then
    echo "Chot cau hinh roi moi chay test (DUNG MOT LAN):"
    echo "  python scripts/08_final_test.py $RUN_DIR --reason \"...\""
fi

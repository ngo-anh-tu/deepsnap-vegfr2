"""
Buoc 3: huan luyen N seed. CHI doc train+val -- KHONG sinh predictions_test.csv (docs §12.6).
Test chay mot lan duy nhat qua 08_final_test.py.

Vi du:  python scripts/03_train.py                                  # v1 + resnet18
        python scripts/03_train.py --config configs/base.yaml configs/render_v1.yaml \
                                   configs/model_r18.yaml configs/ablation/scratch.yaml
"""
import argparse, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, abs_path, run_name
from src.m6_train import train_seed
from src.m7_eval import aggregate_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+',
                    default=['configs/base.yaml', 'configs/render_v1.yaml', 'configs/model_r18.yaml'])
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='chi N phan tu moi split (chay thu)')
    ap.add_argument('--run-name', default=None)
    a = ap.parse_args()
    cfg = load_config(*a.config)
    if a.epochs: cfg['train']['epochs'] = a.epochs
    if a.limit:  cfg['data']['limit_mols'] = a.limit
    seeds = a.seeds if a.seeds is not None else cfg['train']['seeds']

    run_dir = abs_path(cfg['paths']['runs_dir']) / (a.run_name or run_name(cfg))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    for s in seeds:
        m = train_seed(cfg, s, run_dir / ('seed%d' % s))
        print('seed %d: val RMSE=%.3f  train RMSE=%.3f  (epoch %d, %d phan tu)'
              % (s, m['rmse'], m['train_rmse'], m['best_epoch'], m['n_molecules']))
    agg = aggregate_seeds(run_dir)
    print('\n%s  val RMSE = %.3f +- %.3f   train RMSE = %.3f   (%d seed)'
          % (run_dir.name, agg['rmse']['mean'], agg['rmse']['std'], agg['train_rmse']['mean'], agg['n_seeds']))


if __name__ == '__main__':
    main()

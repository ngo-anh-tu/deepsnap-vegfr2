"""
Buoc 8: chay test set DUNG MOT LAN cho mot run da chot (K.7, docs §12.6).
Doc seed*/best.ckpt, ghi seed*/predictions_test.csv + predictions_test.csv (trung binh seed, muc phan tu)
+ test_metrics.json, roi APPEND 1 dong vao reports/test_set_log.md.
Tu choi chay lan hai tren cung run, tru khi --again (va phai ghi ly do moi).
"""
import argparse, json, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.config import abs_path, render_dir, config_sha256, git_sha
from src.m4_dataset import RenderDataset
from src.m5_model import build_model
from src.m6_train import select_device
from src.m7_eval import (predict, regression_metrics, bootstrap_ci, test_already_logged,
                         append_test_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='vd runs/v1_r18')
    ap.add_argument('--reason', required=True, help='ly do chay (ghi vao test_set_log.md)')
    ap.add_argument('--again', action='store_true')
    a = ap.parse_args()
    run_dir = abs_path(a.run_dir)
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    reports = abs_path(cfg['paths']['reports_dir'])
    if test_already_logged(reports, run_dir.name) and not a.again:
        raise SystemExit('%s da cham test roi (xem reports/test_set_log.md). K.7: chi MOT lan.' % run_dir.name)

    dev = select_device()
    ds = RenderDataset(render_dir(cfg), 'test', cfg, train=False, allow_test=True,
                       use_cache=cfg['data'].get('use_cache', False))
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                    num_workers=cfg['train']['num_workers'])
    use_tta = cfg['tta']['enabled'] and 'test' in cfg['tta']['apply_to']
    preds, per_seed = [], []
    for sd in sorted(run_dir.glob('seed*/best.ckpt')):
        ck = torch.load(sd, map_location=dev)
        model = build_model(cfg, ck['y_mean'], ck['y_std']).to(dev)
        model.load_state_dict(ck['model'])
        df = predict(model, dl, dev, cfg, use_tta, cfg['train']['amp'])
        df.insert(1, 'smiles', ds.smiles)
        df.to_csv(sd.parent / 'predictions_test.csv', index=False)
        per_seed.append(dict(seed=ck['seed'], **regression_metrics(df['y_true'], df['y_pred'])))
        preds.append(df)
        print('seed %d: test RMSE = %.3f' % (ck['seed'], per_seed[-1]['rmse']))

    ens = preds[0].copy()
    ens['y_pred'] = np.mean([p['y_pred'].to_numpy() for p in preds], axis=0)
    ens[['smiles', 'y_true', 'y_pred']].to_csv(run_dir / 'predictions_test.csv', index=False)
    m = regression_metrics(ens['y_true'], ens['y_pred'])
    lo, hi = bootstrap_ci(ens['y_true'], ens['y_pred'], 'rmse', cfg['eval']['bootstrap_n'], cfg['eval']['ci'])
    out = dict(run=run_dir.name, git_sha=git_sha(), config_sha256=config_sha256(cfg), split='test',
               n_molecules=int(len(ens)), per_seed=per_seed,
               rmse_seed_mean=float(np.mean([s['rmse'] for s in per_seed])),
               rmse_seed_std=float(np.std([s['rmse'] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
               ensemble=dict(**m, rmse_ci=[lo, hi]))
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(out, f, indent=2); f.write('\n')
    append_test_log(reports, run_dir.name, out['config_sha256'], a.reason)
    print('\nTEST %s: RMSE seed-mean = %.3f +- %.3f | ensemble = %.3f [%.3f, %.3f] (%d phan tu)'
          % (run_dir.name, out['rmse_seed_mean'], out['rmse_seed_std'], m['rmse'], lo, hi, len(ens)))


if __name__ == '__main__':
    main()

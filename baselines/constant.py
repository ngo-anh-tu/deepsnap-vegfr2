"""
Baseline 1: du bao hang so = mean(label train). Do tren split_all.csv (§3: RMSE hang so = 1,091).
Schema chung moi baseline (§12.7): runs/baseline_<ten>/predictions_test.csv voi cot smiles,y_true,y_pred.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from src.config import load_config, abs_path, git_sha
from src.m7_eval import append_test_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml'])
    cfg = load_config(*ap.parse_args().config)
    df = pd.read_csv(abs_path(cfg['paths']['split_csv']))
    lab, smi = cfg['data']['label_col'], cfg['data']['smiles_col']
    tr, va, te = (df[df['split'] == s] for s in ('train', 'val', 'test'))
    c = float(tr[lab].mean())

    out_dir = abs_path(cfg['paths']['runs_dir']) / 'baseline_constant'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'smiles': te[smi].values, 'y_true': te[lab].values, 'y_pred': c}) \
        .to_csv(out_dir / 'predictions_test.csv', index=False)
    # r2/pearson khong xac dinh cho du bao hang so (phuong sai 0) -> chi tinh rmse/mae
    err = lambda d: dict(rmse=float(np.sqrt(((d[lab] - c) ** 2).mean())), mae=float((d[lab] - c).abs().mean()))
    mv, mt = err(va), err(te)
    out = dict(baseline='constant', git_sha=git_sha(), constant=c, n_molecules=int(len(te)),
               val=dict(rmse=mv['rmse'], mae=mv['mae']), test=dict(rmse=mt['rmse'], mae=mt['mae']))
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2); f.write('\n')
    append_test_log(abs_path(cfg['paths']['reports_dir']), 'baseline_constant', '-', 'baseline hang so (§12.7)')
    print('constant = %.4f | test RMSE = %.3f (ky vong 1,091) | val RMSE = %.3f | N test = %d'
          % (c, mt['rmse'], mv['rmse'], len(te)))


if __name__ == '__main__':
    main()

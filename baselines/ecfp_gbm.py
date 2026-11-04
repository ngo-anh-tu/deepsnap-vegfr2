"""
Baseline 2: ECFP4 (2048 bit) + descriptor RDKit 2D -> gradient boosting, 5 seed khoi tao.
Chay tren split_all.csv (KHONG dung so 0,751 / 0,674 cu: khac split, §12.7).
Can cho bootstrap ghep cap o M8: y_pred muc phan tu cua mot baseline to-po tren DUNG 932 chat test.
Ra: runs/baseline_ecfp_gbm/{predictions_test.csv, metrics.json}. predictions = trung binh 5 seed.
Cache features: data/interim/ecfp_features.npz (khoa = hash SMILES+split + NBITS + descriptor + phien ban RDKit;
lech khoa -> tu tinh lai). --no-cache: bo qua cache. --featurize-only: chi build cache, KHONG cham test.
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm
from src.config import load_config, abs_path, git_sha
from src.m7_eval import regression_metrics, append_test_log
RDLogger.DisableLog('rdApp.*')

SEEDS = (0, 1, 2, 3, 4)
NBITS = 2048
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)


def featurize(smiles):
    X = []
    for s in tqdm(smiles, desc='featurize', leave=False):
        m = Chem.MolFromSmiles(s)
        fp = np.asarray(_gen.GetFingerprintAsNumPy(m), dtype=np.float32)
        d = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        d[~np.isfinite(d)] = np.nan
        X.append(np.concatenate([fp, np.clip(d, -1e6, 1e6).astype(np.float32)]))
    return np.stack(X)


def cache_key(parts, smi):
    h = hashlib.sha256()
    for s in ('train', 'val', 'test'):
        h.update(s.encode()); h.update('\n'.join(parts[s][smi]).encode())
    h.update(('nbits=%d|radius=2|rdkit=%s|desc=%s' % (NBITS, rdkit.__version__,
              ','.join(n for n, _ in Descriptors.descList))).encode())
    return h.hexdigest()


def load_or_featurize(parts, smi, path, use_cache=True):
    key = cache_key(parts, smi)
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z['key']) == key:
            print('features: doc cache %s' % path)
            return {s: z[s] for s in ('train', 'val', 'test')}
        print('features: cache lech khoa -> tinh lai')
    X = {s: featurize(p[smi]) for s, p in parts.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, key=np.array(key), **X)
    tmp.replace(path)
    print('features: da luu cache %s' % path)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml'])
    ap.add_argument('--no-cache', action='store_true', help='bo qua cache features, tinh lai')
    ap.add_argument('--featurize-only', action='store_true', help='chi build cache features roi thoat')
    args = ap.parse_args()
    cfg = load_config(*args.config)
    df = pd.read_csv(abs_path(cfg['paths']['split_csv']))
    lab, smi = cfg['data']['label_col'], cfg['data']['smiles_col']
    parts = {s: df[df['split'] == s].reset_index(drop=True) for s in ('train', 'val', 'test')}
    cache_path = abs_path(cfg['paths']['interim_dir']) / 'ecfp_features.npz'
    X = load_or_featurize(parts, smi, cache_path, use_cache=not args.no_cache)
    if args.featurize_only:
        print('shape: ' + ', '.join('%s=%s' % (k, v.shape) for k, v in X.items()))
        return

    pv, pt, per_seed = [], [], []
    for sd in SEEDS:
        # max_features<1 -> seed co tac dung; sieu tham so co dinh, KHONG tinh chinh tren test
        mdl = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_features=0.3,
                                            random_state=sd)
        mdl.fit(X['train'], parts['train'][lab])
        pv.append(mdl.predict(X['val'])); pt.append(mdl.predict(X['test']))
        per_seed.append(dict(seed=sd, val_rmse=regression_metrics(parts['val'][lab], pv[-1])['rmse'],
                             test_rmse=regression_metrics(parts['test'][lab], pt[-1])['rmse']))
        print('seed %d: val RMSE = %.3f | test RMSE = %.3f' % (sd, per_seed[-1]['val_rmse'], per_seed[-1]['test_rmse']))

    te = parts['test']
    ens = np.mean(pt, axis=0)
    out_dir = abs_path(cfg['paths']['runs_dir']) / 'baseline_ecfp_gbm'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'smiles': te[smi], 'y_true': te[lab], 'y_pred': ens}).to_csv(out_dir / 'predictions_test.csv', index=False)
    r = [s['test_rmse'] for s in per_seed]
    out = dict(baseline='ecfp_gbm', git_sha=git_sha(), n_molecules=int(len(te)), per_seed=per_seed,
               test_rmse_seed_mean=float(np.mean(r)), test_rmse_seed_std=float(np.std(r, ddof=1)),
               val=regression_metrics(parts['val'][lab], np.mean(pv, axis=0)),
               test=regression_metrics(te[lab], ens))
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2); f.write('\n')
    append_test_log(abs_path(cfg['paths']['reports_dir']), 'baseline_ecfp_gbm', '-', 'baseline to-po ECFP4+descriptor (§12.7)')
    print('\ntest RMSE: seed-mean %.3f +- %.3f | ensemble %.3f | N=%d'
          % (out['test_rmse_seed_mean'], out['test_rmse_seed_std'], out['test']['rmse'], len(te)))


if __name__ == '__main__':
    main()

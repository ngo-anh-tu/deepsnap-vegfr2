"""
Buoc 11: gom ket qua MOI run trong runs/ thanh mot bang de so sanh (train / val / test).
  CNN run : doc <run>/metrics_all_splits.csv (10_split_metrics.py). Run chua co file do (dang train)
            -> chi tinh val tu seed*/predictions_val.csv da co, danh dau 'dang chay'.
  baseline: doc <run>/metrics.json (chi co val + test).
Script nay KHONG mo split test: test chi co neu 08_final_test.py da cham run do (K.7).
Ra: reports/summary_runs.csv + reports/summary_runs.md
Vi du:  python scripts/11_summary_runs.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from src.config import abs_path
from src.m7_eval import regression_metrics

KEYS = ['rmse', 'mae', 'r2', 'pearson', 'spearman']


def cnn_rows(run_dir):
    f = run_dir / 'metrics_all_splits.csv'
    if f.exists():
        d = pd.read_csv(f)
        n_seed = int((~d['seed'].isin(['mean', 'std', 'ensemble']) & (d['split'] == 'val')).sum())
        for split, g in d.groupby('split', sort=False):
            g = g.set_index('seed')
            row = dict(run=run_dir.name, split=split, status='xong', n_seed=n_seed,
                       n_molecules=int(g.loc['mean', 'n_molecules']))
            for k in KEYS:
                row[k + '_mean'] = float(g.loc['mean', k])
                row[k + '_std'] = float(g.loc['std', k])
                row[k + '_ens'] = float(g.loc['ensemble', k])
            yield row
        return
    preds = [pd.read_csv(p) for p in sorted(run_dir.glob('seed*/predictions_val.csv'))]
    if not preds:
        return
    per = [regression_metrics(p['y_true'], p['y_pred']) for p in preds]
    ens = regression_metrics(preds[0]['y_true'], np.mean([p['y_pred'].to_numpy() for p in preds], axis=0))
    row = dict(run=run_dir.name, split='val', status='dang chay', n_seed=len(preds), n_molecules=len(preds[0]))
    for k in KEYS:
        v = [m[k] for m in per]
        row[k + '_mean'] = float(np.mean(v))
        row[k + '_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
        row[k + '_ens'] = float(ens[k])
    yield row


def baseline_rows(run_dir):
    m = json.load(open(run_dir / 'metrics.json'))
    for split in ('val', 'test'):
        if split not in m:
            continue
        row = dict(run=run_dir.name, split=split, status='xong', n_seed=len(m.get('per_seed', [])) or 1,
                   n_molecules=m.get('n_molecules'))
        for k in KEYS:
            if k in m[split]:
                row[k + '_ens'] = float(m[split][k])
        yield row


def to_md(df):
    def fmt(r, k):
        if pd.isna(r.get(k + '_mean')):
            return '%.3f' % r[k + '_ens'] if not pd.isna(r.get(k + '_ens')) else '-'
        sd = '' if pd.isna(r[k + '_std']) else ' ± %.3f' % r[k + '_std']
        return '%.3f%s' % (r[k + '_mean'], sd)
    lines = ['# Tong hop ket qua moi run', '',
             'Sinh boi `scripts/11_summary_runs.py` -- khong chep tay. CNN: trung binh ± std qua seed, '
             'cot `ens` = trung binh du doan cac seed. Baseline: mot gia tri. '
             'Test chi co khi run da qua `08_final_test.py` (K.7).', '']
    for split in ('train', 'val', 'test'):
        s = df[df['split'] == split]
        if s.empty:
            continue
        lines += ['## %s' % split, '',
                  '| run | trang thai | seed | RMSE | RMSE ens | MAE | R² | R² ens | Pearson | Spearman |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for _, r in s.sort_values('rmse_ens').iterrows():
            lines.append('| %s | %s | %s | %s | %.3f | %s | %s | %s | %s | %s |' % (
                r['run'], r['status'], r['n_seed'], fmt(r, 'rmse'), r['rmse_ens'], fmt(r, 'mae'),
                fmt(r, 'r2'), '-' if pd.isna(r.get('r2_ens')) else '%.3f' % r['r2_ens'],
                fmt(r, 'pearson'), fmt(r, 'spearman')))
        lines.append('')
    return '\n'.join(lines)


def main():
    runs = abs_path('runs')
    rows = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        rows += list(baseline_rows(d) if d.name.startswith('baseline_') else cnn_rows(d))
    df = pd.DataFrame(rows)
    out = abs_path('reports')
    df.to_csv(out / 'summary_runs.csv', index=False)
    md = to_md(df)
    (out / 'summary_runs.md').write_text(md + '\n')
    print(md)
    print('-> %s/summary_runs.{csv,md}' % out)


if __name__ == '__main__':
    main()

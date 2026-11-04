"""
Buoc 10: gom chi so hoi quy + phan loai cho train / val / test cua MOT run vao mot file.
  train: du bao lai tu seed*/best.ckpt (khong augment, TTA nhu luc do train_rmse) -> seed*/predictions_train.csv
  val  : doc seed*/predictions_val.csv (03_train.py da ghi)
  test : CHI doc seed*/predictions_test.csv neu 08_final_test.py da chay -- script nay KHONG mo split test (K.7).
Ra: <run>/metrics_all_splits.csv (moi dong = split x seed, kem 'mean', 'std', 'ensemble') + .json
Vi du:  python scripts/10_split_metrics.py runs/v1_r18
"""
import argparse, json, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.config import abs_path, render_dir
from src.m4_dataset import RenderDataset
from src.m5_model import build_model
from src.m6_train import select_device
from src.m7_eval import predict, regression_metrics, classification_metrics

THRESHOLDS = (6.0, 7.0)      # pIC50 >= 6: IC50 <= 1 uM ; >= 7: IC50 <= 100 nM


def all_metrics(y, p) -> dict:
    out = regression_metrics(y, p)
    for t in THRESHOLDS:
        out.update(classification_metrics(y, p, t))
    return out


def predict_train(run_dir, cfg, ckpts):
    dev = select_device()
    ds = RenderDataset(render_dir(cfg), 'train', cfg, train=False, use_cache=cfg['data'].get('use_cache', False))
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                    num_workers=cfg['train']['num_workers'])
    use_tta = cfg['tta']['enabled'] and 'val' in cfg['tta']['apply_to']     # giong m6_train khi do train_rmse
    for ck_path in ckpts:
        out = ck_path.parent / 'predictions_train.csv'
        if out.exists():
            continue
        ck = torch.load(ck_path, map_location=dev)
        model = build_model(cfg, ck['y_mean'], ck['y_std']).to(dev)
        model.load_state_dict(ck['model'])
        df = predict(model, dl, dev, cfg, use_tta, cfg['train']['amp'])
        df.insert(1, 'smiles', ds.smiles)
        df.to_csv(out, index=False)
        print('seed %d: -> %s' % (ck['seed'], out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', help='vd runs/v1_r18')
    a = ap.parse_args()
    run_dir = abs_path(a.run_dir)
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    ckpts = sorted(run_dir.glob('seed*/best.ckpt'))
    predict_train(run_dir, cfg, ckpts)

    rows = []
    for split in ('train', 'val', 'test'):
        files = sorted(run_dir.glob('seed*/predictions_%s.csv' % split))
        if not files:
            print('%s: chua co predictions (test -> chay 08_final_test.py truoc)' % split)
            continue
        preds = []
        for f in files:
            df = pd.read_csv(f).sort_values('mol_idx').reset_index(drop=True)
            rows.append(dict(split=split, seed=f.parent.name.replace('seed', ''), n_molecules=len(df),
                             **all_metrics(df['y_true'], df['y_pred'])))
            preds.append(df)
        assert all((p['mol_idx'].to_numpy() == preds[0]['mol_idx'].to_numpy()).all() for p in preds)
        per_seed = pd.DataFrame(rows[-len(files):]).drop(columns=['split', 'seed'])
        rows.append(dict(split=split, seed='mean', **per_seed.mean().to_dict()))
        rows.append(dict(split=split, seed='std', **per_seed.std(ddof=1).to_dict()))
        ens = np.mean([p['y_pred'].to_numpy() for p in preds], axis=0)
        rows.append(dict(split=split, seed='ensemble', n_molecules=len(ens),
                         **all_metrics(preds[0]['y_true'], ens)))

    res = pd.DataFrame(rows)
    res['n_molecules'] = res['n_molecules'].astype(int)
    res.to_csv(run_dir / 'metrics_all_splits.csv', index=False)
    with open(run_dir / 'metrics_all_splits.json', 'w') as f:
        json.dump(dict(run=run_dir.name, thresholds=list(THRESHOLDS), rows=res.to_dict('records')), f, indent=2)
        f.write('\n')
    cols = ['split', 'seed', 'n_molecules', 'rmse', 'mae', 'r2', 'pearson', 'spearman',
            'acc@6', 'bal_acc@6', 'mcc@6', 'auc@6', 'acc@7', 'bal_acc@7', 'mcc@7', 'auc@7']
    print(res[res['seed'].isin(['mean', 'std', 'ensemble'])][cols].to_string(index=False, float_format='%.3f'))
    print('-> %s' % (run_dir / 'metrics_all_splits.csv'))


if __name__ == '__main__':
    main()

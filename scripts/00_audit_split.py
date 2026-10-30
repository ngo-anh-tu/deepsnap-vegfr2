"""
Audit split_all.csv -> reports/split_audit.md (docs §12.5: chinh bang §3, chay duoc chu khong chep tay).

Kiem: scaffold xuyen split, lech mean test-train, RMSE hang so, quy uoc scaffold
(MurckoScaffoldSmiles(includeChirality=True)), cap enantiomer bi tach.
Muc "35 chat bi loai so voi 9.341" can file clean (--clean csv co cot smiles) -- khong co trong repo.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from src.config import load_config, abs_path
RDLogger.DisableLog('rdApp.*')

EXPECT = dict(scaffold_cross=0, n_scaffold=3521, shift=-0.063, rmse_const=1.091, scaf_match=100.0,
              dropped=35, enant_pairs=75, enant_split=10, enant_train_test=3, n_test=932)


def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=True) if m else None


def mirror_canon(smi):
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(m)


def enantiomer_pairs(df, smi_col):
    canon = {i: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for i, s in zip(df.index, df[smi_col])}
    by = {c: i for i, c in canon.items()}
    pairs = set()
    for i, s in zip(df.index, df[smi_col]):
        if canon[i] == mirror_canon(s):         # khong co tam lap the -> khong co doi quang
            continue
        j = by.get(mirror_canon(s))
        if j is not None and j != i:
            pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def row(name, got, want, fmt, tol):
    ok = got is not None and abs(got - want) <= tol
    return '| %s | %s | %s | %s |' % (name, fmt % got if got is not None else 'n/a', fmt % want,
                                       'OK' if ok else 'LECH')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', default=None, help='csv 9.341 chat sau clean (cot smiles) de dem chat bi loai')
    a = ap.parse_args()
    cfg = load_config('configs/base.yaml')
    lab, smi = cfg['data']['label_col'], cfg['data']['smiles_col']
    df = pd.read_csv(abs_path(cfg['paths']['split_csv']))
    tr, te = df[df['split'] == 'train'], df[df['split'] == 'test']

    sp = df.groupby('scaffold')['split'].nunique()
    cross = int((sp > 1).sum())
    shift = float(te[lab].mean() - tr[lab].mean())
    rmse = float(np.sqrt(((te[lab] - tr[lab].mean()) ** 2).mean()))
    recomputed = df[smi].map(murcko)
    match = 100.0 * float((recomputed == df['scaffold']).mean())

    pairs = enantiomer_pairs(df, smi)
    sp_pairs = [(i, j) for i, j in pairs if df.at[i, 'split'] != df.at[j, 'split']]
    tt = [(i, j) for i, j in sp_pairs if {df.at[i, 'split'], df.at[j, 'split']} == {'train', 'test'}]

    dropped = None
    if a.clean:
        clean = pd.read_csv(a.clean)
        dropped = int(len(set(clean[smi]) - set(df[smi])))

    L = ['# Audit split_all.csv', '',
         'Sinh boi `scripts/00_audit_split.py` -- khong chep tay. "LECH" = khac so ghi o §3.', '',
         '| Muc | Do duoc | Ky vong (§3) | |', '|---|---|---|---|',
         row('scaffold xuyen split', cross, EXPECT['scaffold_cross'], '%d', 0),
         row('so scaffold', int(df['scaffold'].nunique()), EXPECT['n_scaffold'], '%d', 0),
         row('lech mean test-train', shift, EXPECT['shift'], '%.3f', 0.0015),
         row('RMSE hang so (mean train)', rmse, EXPECT['rmse_const'], '%.3f', 0.0015),
         row('scaffold khop quy uoc includeChirality=True (%)', match, EXPECT['scaf_match'], '%.1f', 0.05),
         row('so chat test', int(len(te)), EXPECT['n_test'], '%d', 0),
         row('chat bi loai so voi 9.341', dropped, EXPECT['dropped'], '%d', 0) if dropped is not None
         else '| chat bi loai so voi 9.341 | n/a (can --clean) | %d | chua kiem |' % EXPECT['dropped'],
         row('cap enantiomer', len(pairs), EXPECT['enant_pairs'], '%d', 0),
         row('  - bi tach qua split', len(sp_pairs), EXPECT['enant_split'], '%d', 0),
         row('  - trong do train<->test', len(tt), EXPECT['enant_train_test'], '%d', 0), '',
         'Phan bo split: ' + ', '.join('%s=%d' % kv for kv in df['split'].value_counts().items()), '']
    out = abs_path(cfg['paths']['reports_dir']) / 'split_audit.md'
    out.parent.mkdir(exist_ok=True)
    out.write_text('\n'.join(L))
    print('\n'.join(L)); print('-> %s' % out)


if __name__ == '__main__':
    main()

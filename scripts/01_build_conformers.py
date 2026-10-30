"""
Buoc 1: SMILES -> conformer 3D (M1) + do hinh hoc sau can chinh (M2).

Sinh ra ca 3 file data/interim/ trong MOT lan chay (docs §12.4):
  conformers.sdf          conformer GOC (chua can chinh, co H). M2 chay lai luc render: tat dinh, re.
  conformers_meta.parquet truy vet M1 (seed, nang luong MMFF, ly do that bai)
  geometry.parquet        flat_ratio, has_undefined_stereo, ext1-3, ... -- COMMIT file nay.
                          flat_ratio va has_undefined_stereo phai co TRUOC khi train (dang ky truoc, §8.3)
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm
from src.config import load_config, abs_path
from src.m1_conformer import embed
from src.m2_align import align
RDLogger.DisableLog('rdApp.*')
PT = Chem.GetPeriodicTable()


def has_undefined_stereo(smiles: str) -> bool:
    """True neu con tam lap the chua xac dinh ('?'). 633/9.341 chat (6,78%) o §2.6; 629 tren 9.306."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    cs = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    return any(tag == '?' for _, tag in cs)


def diameter_vdw(m: Chem.Mol) -> float:
    P = m.GetConformer().GetPositions()
    r = np.array([PT.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return float((D + r[:, None] + r[None, :]).max())


def work(task):
    idx, smi = task
    row = dict(orig_idx=idx, smiles=smi, embed_ok=False, embed_seed=None, mmff_energy=None,
               reason='', has_undefined_stereo=has_undefined_stereo(smi))
    res = embed(smi)
    row['reason'] = res.reason
    if not res.ok:
        return row, None
    m = res.mol
    am, info = align(m)
    heavy = [a.GetIdx() for a in am.GetAtoms() if a.GetAtomicNum() > 1]
    Ph = am.GetConformer().GetPositions()[heavy]
    row.update(embed_ok=True, embed_seed=res.seed_used, mmff_energy=res.mmff_energy,
               n_heavy=len(heavy), n_atoms=am.GetNumAtoms(),
               diameter_vdw=diameter_vdw(am),
               radius_gyration=float(np.sqrt(((Ph - Ph.mean(0)) ** 2).sum(1).mean())),
               **{k: info[k] for k in ('ext1', 'ext2', 'ext3', 'flat_ratio', 'axes_degenerate')})
    m.SetProp('_Name', str(idx))
    block = Chem.MolToMolBlock(m) + '> <orig_idx>\n%d\n\n> <smiles>\n%s\n\n$$$$\n' % (idx, smi)
    return row, block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='chi chay N chat dau (thu nhanh)')
    a = ap.parse_args()
    cfg = load_config(*a.config)

    df = pd.read_csv(abs_path(cfg['paths']['split_csv']))
    limit = a.limit or cfg['data']['limit_mols']
    if limit:
        df = df.head(limit)
    tasks = list(zip(df[cfg['data']['id_col']], df[cfg['data']['smiles_col']]))

    with Pool(a.workers) as pool:
        out = list(tqdm(pool.imap(work, tasks, chunksize=16), total=len(tasks), desc='conformer'))

    rows = [r for r, _ in out]
    g = pd.DataFrame(rows).sort_values('orig_idx').reset_index(drop=True)
    out_dir = abs_path(cfg['paths']['interim_dir']); out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'conformers.sdf', 'w') as f:
        for _, block in sorted(((r['orig_idx'], b) for r, b in out if b), key=lambda t: t[0]):
            f.write(block)
    g[['orig_idx', 'smiles', 'embed_ok', 'embed_seed', 'mmff_energy', 'reason']] \
        .to_parquet(out_dir / 'conformers_meta.parquet', index=False)
    cols = ['orig_idx', 'flat_ratio', 'has_undefined_stereo', 'diameter_vdw', 'radius_gyration',
            'ext1', 'ext2', 'ext3', 'axes_degenerate', 'n_heavy', 'n_atoms', 'embed_seed', 'embed_ok']
    g[cols].to_parquet(out_dir / 'geometry.parquet', index=False)

    ok = g['embed_ok']
    thr = cfg['geometry']['flat_ratio_threshold']
    print('\nembed OK      : %d / %d  (that bai: %d)' % (ok.sum(), len(g), (~ok).sum()))
    print('stereo chua xd: %d (%.2f%%)   [§2.6: 633/9.341 = 6,78%%; tren 9.306 chat do duoc 629 = 6,76%%]'
          % (g['has_undefined_stereo'].sum(), 100 * g['has_undefined_stereo'].mean()))
    print('flat_ratio<%.2f: %d (%.2f%%)' % (thr, (g['flat_ratio'] < thr).sum(),
                                         100 * (g['flat_ratio'] < thr).sum() / max(ok.sum(), 1)))
    print('truc suy bien : %d (%.2f%%)  [ky vong ~0,08%% o §4]'
          % (g['axes_degenerate'].fillna(False).sum(), 100 * g['axes_degenerate'].fillna(False).mean()))
    print('diameter_vdw max = %.2f A' % g['diameter_vdw'].max())
    if (~ok).any():
        print('\nChat BI BO (ghi log, khong bo im lang):')
        print(g.loc[~ok, ['orig_idx', 'smiles', 'reason']].to_string(index=False))
    print('\nviet -> %s' % out_dir)


if __name__ == '__main__':
    main()

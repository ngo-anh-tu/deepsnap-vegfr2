"""Phep thu #2: chat lon nhat khong bi cat o BAT KY goc nao + do throughput that."""
import numpy as np, os, sys, time, tempfile, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from PIL import Image
from src.m3_render import Renderer, VDW_SCALE
RDLogger.DisableLog('rdApp.*')
SPLIT = os.path.join(ROOT, 'data', 'raw', 'split_all.csv')

def prep(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    for sd in (0xf00d, 0xbeef, 0x1234):
        ps = AllChem.ETKDGv3(); ps.randomSeed = sd; ps.useRandomCoords = True
        if AllChem.EmbedMolecule(m, ps) >= 0: break
    else: return None
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=800)
    except Exception: pass
    conf = m.GetConformer(); P = conf.GetPositions()
    hv = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    c = P[hv].mean(0)
    U,S,Vt = np.linalg.svd(P[hv]-c, full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Q = (P-c) @ Vt.T
    for i in range(m.GetNumAtoms()): conf.SetAtomPosition(i, Point3D(*Q[i]))
    return m

def touches_border(f, pad=1):
    a = np.array(Image.open(f).convert('L'))
    b = a < 250
    return bool(b[:pad,:].any() or b[-pad:,:].any() or b[:,:pad].any() or b[:,-pad:].any())

def rand_rot(rng):
    A = rng.normal(size=(3,3)); Q,_ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0: Q[:,2] *= -1
    return Q

def main():
    clean = pd.read_csv(SPLIT).rename(columns={'smiles': 'can'})
    clean['heavy'] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in clean['can']]
    big = clean.nlargest(6, 'heavy')
    tmp = tempfile.mkdtemp(); R = Renderer(); rng = np.random.default_rng(0)
    print('camera_z = %.2f   px/A = %.3f' % (R.calib['z'], R.calib['px_per_a_measured']))
    print()
    print('--- 6 chat LON NHAT, moi chat 12 goc ngau nhien ---')
    bad = 0
    for _, row in big.iterrows():
        m = prep(row['can'])
        if m is None: print('  embed that bai'); continue
        hits = 0
        for k in range(12):
            f = R.render(m, rot=rand_rot(rng), out=os.path.join(tmp,'b.png'))
            if touches_border(f): hits += 1
        bad += hits
        print('  %3d nguyen tu nang : %2d/12 goc bi cat' % (row['heavy'], hits))
    print()
    print('--- throughput (100 chat ngau nhien, 1 anh moi chat) ---')
    smp = clean['can'].sample(100, random_state=7).tolist()
    mols = [prep(s) for s in smp]; mols = [m for m in mols if m is not None]
    t0 = time.time()
    for i, m in enumerate(mols):
        R.render(m, out=os.path.join(tmp,'t.png'))
    dt = (time.time()-t0)/len(mols)
    print('  %.1f ms/anh  ->  V1 (9.306 anh) = %.1f phut | V27 (251.262 anh) = %.1f gio'
          % (1000*dt, 9306*dt/60, 251262*dt/3600))
    print()
    print('[%s] khong chat nao bi cat o bat ky goc nao' % ('PASS' if bad==0 else 'FAIL (%d lan)'%bad))
    R.close()
    return 0 if bad==0 else 1

if __name__ == '__main__':
    raise SystemExit(main())


def test_frame_fits():
    import pytest
    pytest.importorskip('pymol2')
    assert main() == 0

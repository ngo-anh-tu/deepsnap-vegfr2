"""Phep thu chan loi im lang #1: thang PHAI tuyet doi, khong fit-to-frame."""
import numpy as np, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from PIL import Image
from src.m3_render import Renderer, VDW_SCALE
RDLogger.DisableLog('rdApp.*')
PT = Chem.GetPeriodicTable()

BENZENE  = 'c1ccccc1'
SORAFENIB= 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1'

def prep(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    ps = AllChem.ETKDGv3(); ps.randomSeed=0xf00d; ps.useRandomCoords=True
    assert AllChem.EmbedMolecule(m, ps) >= 0
    AllChem.MMFFOptimizeMolecule(m, maxIters=800)
    conf = m.GetConformer(); P = conf.GetPositions()
    hv = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    c = P[hv].mean(0)
    U,S,Vt = np.linalg.svd(P[hv]-c, full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Q = (P - c) @ Vt.T
    from rdkit.Geometry import Point3D
    for i in range(m.GetNumAtoms()): conf.SetAtomPosition(i, Point3D(*Q[i]))
    r = np.array([PT.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()]) * VDW_SCALE
    # hop bao cua phep chieu x-y (sau can chinh) -- cung dinh nghia voi _probe_extent_px
    ext_proj = float(max((Q[:,0]+r).max()-(Q[:,0]-r).min(),
                         (Q[:,1]+r).max()-(Q[:,1]-r).min()))
    D = np.linalg.norm(Q[:,None,:]-Q[None,:,:], axis=-1)
    ext_3d = float((D + r[:,None] + r[None,:]).max())
    return m, ext_proj, ext_3d

def px_extent(f):
    a = np.array(Image.open(f).convert('L'))
    nz = np.argwhere(a < 250)
    return float(max(np.ptp(nz[:,0]), np.ptp(nz[:,1])) + 1) if len(nz) else 0.0

def main():
    tmp = tempfile.mkdtemp(); R = Renderer()
    print('hieu chuan: k=%.1f  camera_z=%.2f  px/A muc tieu=%.3f  do lai=%.3f'
          % (R.calib['k'], R.calib['z'], R.calib['px_per_a_target'], R.calib['px_per_a_measured']))
    res = {}
    for nm, smi in [('benzene', BENZENE), ('sorafenib', SORAFENIB)]:
        m, ext, ext3d = prep(smi)
        f = R.render(m, out=os.path.join(tmp, nm + '.png'))
        px = px_extent(f)
        res[nm] = (ext, px, px/ext)
        print('%-10s chieu=%6.2f A  (3D=%6.2f A)  pixel=%5.0f px  => %.3f px/A'
              % (nm, ext, ext3d, px, px/ext))
    rA = res['sorafenib'][0] / res['benzene'][0]
    rP = res['sorafenib'][1] / res['benzene'][1]
    print()
    print('ty le kich thuoc THUC      = %.3f' % rA)
    print('ty le kich thuoc PIXEL     = %.3f' % rP)
    print('lech                       = %.1f%%' % (100*abs(rP-rA)/rA))
    ok_scale = abs(rP - rA) / rA < 0.08
    ok_fit   = abs(rP - 1.0) > 0.25          # neu ~1.0 thi da thanh fit-to-frame
    ok_px    = all(abs(v[2] - R.px_per_a) / R.px_per_a < 0.08 for v in res.values())
    print()
    print('[%s] ty le pixel khop ty le thuc' % ('PASS' if ok_scale else 'FAIL'))
    print('[%s] KHONG phai fit-to-frame'      % ('PASS' if ok_fit   else 'FAIL'))
    print('[%s] px/A dung muc tieu 8.0'       % ('PASS' if ok_px    else 'FAIL'))
    R.close()
    return 0 if (ok_scale and ok_fit and ok_px) else 1

if __name__ == '__main__':
    raise SystemExit(main())


def test_scale_absolute():
    import pytest
    pytest.importorskip('pymol2')
    assert main() == 0

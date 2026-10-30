"""Pipeline khong duoc tao anh guong: lat guong = doi R <-> S, pha huy nhan."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools, numpy as np, pandas as pd, pytest, torch
from rdkit import Chem
from src.config import load_config
from src.m1_conformer import embed
from src.m2_align import align
from src.m4_dataset import affine_theta, tta_thetas, sample_train_theta
from src.views import view_rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    return load_config('configs/base.yaml', 'configs/render_v1.yaml')


def test_every_view_is_a_proper_rotation():
    for i, j in itertools.permutations(range(3), 2):
        assert abs(np.linalg.det(view_rotation((i, j))) - 1.0) < 1e-9


def test_tta_transforms_are_not_reflections():
    for th in tta_thetas(_cfg()):
        assert float(torch.det(th[:, :2])) > 0.99


def test_train_augment_is_not_reflection():
    cfg = _cfg(); torch.manual_seed(0)
    for _ in range(500):
        assert float(torch.det(sample_train_theta(cfg)[:, :2])) > 0.99
    assert float(torch.det(affine_theta(0, 0, 0, 256)[:, :2])) == pytest.approx(1.0)


def test_no_flip_ops_in_source():
    for f in glob.glob(os.path.join(ROOT, 'src', '*.py')) + glob.glob(os.path.join(ROOT, 'scripts', '*.py')):
        txt = open(f).read().lower()
        assert 'randomhorizontalflip' not in txt and 'hflip' not in txt and '.flip(' not in txt, f


def _cip(m):
    m = Chem.Mol(m)
    Chem.AssignStereochemistryFrom3D(m)
    return {i: t for i, t in Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)}


def test_m2_preserves_cip_labels():
    path = os.path.join(ROOT, 'data', 'raw', 'split_all.csv')
    if not os.path.exists(path):
        pytest.skip('thieu split_all.csv')
    smi = [s for s in pd.read_csv(path)['smiles'] if '@' in s][:40]
    assert len(smi) >= 20
    n = 0
    for s in smi:
        r = embed(s)
        if not r.ok:
            continue
        before = _cip(r.mol)
        aligned, _ = align(r.mol)
        assert _cip(aligned) == before, s
        n += 1
    assert n >= 20

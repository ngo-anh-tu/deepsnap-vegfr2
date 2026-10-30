"""Phep thu chan loi im lang #3 (§12.3): M2 phai TAT DINH, ke ca khi mo men bac ba ~ 0 (chat doi xung cao)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pytest
from rdkit import Chem
from src.m1_conformer import embed
from src.m2_align import align

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMMETRIC = {
    'benzene': 'c1ccccc1',
    'adamantane': 'C1C2CC3CC1CC(C2)C3',
    "4,4'-bipyridine": 'c1cnccc1-c1ccncc1',
}


def coords(m):
    return m.GetConformer().GetPositions()


def test_same_input_twice_is_bitwise_identical():
    for name, smi in SYMMETRIC.items():
        m = embed(smi).mol
        a, _ = align(m)
        b, _ = align(m)
        assert np.array_equal(coords(a), coords(b)), name


def test_fresh_embed_then_align_is_reproducible():
    # cung seed ETKDG -> cung conformer -> cung anh giua cac lan chay
    for name, smi in SYMMETRIC.items():
        a, _ = align(embed(smi).mol)
        b, _ = align(embed(smi).mol)
        assert np.array_equal(coords(a), coords(b)), name


def test_symmetric_molecules_hit_the_tiebreak_path():
    # it nhat mot chat doi xung phai that su la truong hop suy bien, neu khong test nay vo nghia
    flags = [align(embed(s).mol)[1]['axes_degenerate'] for s in SYMMETRIC.values()]
    assert any(flags)


def test_rotation_not_reflection_on_sample():
    path = os.path.join(ROOT, 'data', 'raw', 'split_all.csv')
    if not os.path.exists(path):
        pytest.skip('thieu split_all.csv')
    smi = pd.read_csv(path)['smiles'].sample(60, random_state=0)
    n = 0
    for s in smi:
        r = embed(s)
        if not r.ok:
            continue
        _, info = align(r.mol)
        assert abs(info['det'] - 1.0) < 1e-6, s
        n += 1
    assert n >= 50

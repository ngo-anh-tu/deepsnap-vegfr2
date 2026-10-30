"""
Goc nhin = phep quay (det=+1) cua he truc da can chinh (M2), KHONG phai phan chieu.
View axes=[i, j]: truc camera x = truc i, y = truc j, nhin doc theo i x j.
  [0,1] -> PC1-PC2 (V1)   [0,2] -> PC1-PC3   [1,2] -> PC2-PC3
View roll_deg=t: truc x = PC1 (giu nguyen do dai), y = cos t PC2 + sin t PC3; t=0 trung PC1-PC2.
"""
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


def view_rotation(axes) -> np.ndarray:
    i, j = axes
    e = np.eye(3)
    R = np.stack([e[i], e[j], np.cross(e[i], e[j])])
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, 'goc nhin phai la phep quay'
    return R


def roll_rotation(deg: float) -> np.ndarray:
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, 'goc nhin phai la phep quay'
    return R


def view_matrix(v: dict) -> np.ndarray:
    """v tu config: {axes: [i, j]} hoac {roll_deg: t}."""
    return roll_rotation(v['roll_deg']) if 'roll_deg' in v else view_rotation(v['axes'])


def rotate_mol(mol: Chem.Mol, R: np.ndarray) -> Chem.Mol:
    """Tra ve ban sao da quay toa do: Q = P @ R.T (cung quy uoc voi m2_align)."""
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, 'R phai co det=+1 (khong phan chieu!)'
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    Q = conf.GetPositions() @ R.T
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*Q[i]))
    return m

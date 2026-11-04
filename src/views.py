"""
Goc nhin = phep quay (det=+1) cua he truc da can chinh (M2), KHONG phai phan chieu.
View axes=[i, j]: truc camera x = truc i, y = truc j, nhin doc theo i x j.
  [0,1] -> PC1-PC2 (V1)   [0,2] -> PC1-PC3   [1,2] -> PC2-PC3
tilt (tuy chon): [[truc_camera, do], ...] quay them quanh truc x/y CUA CAMERA, ap theo thu tu.
  axes [0,1] + tilt [[x, 45]] -> nhin doc (PC2+PC3)/sqrt2 ; [[y, 45]] -> (PC3-PC1)/sqrt2  (V9)
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


def camera_rotation(axis: str, deg: float) -> np.ndarray:
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    raise ValueError('tilt chi quanh truc camera x hoac y, nhan %r' % axis)


def view_matrix(view: dict) -> np.ndarray:
    """view: mot muc cua render.views ({name, axes, tilt?}). Tilt quay trong he camera: R = T @ R_axes."""
    R = view_rotation(view['axes'])
    for axis, deg in view.get('tilt', []):
        R = camera_rotation(axis, deg) @ R
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, 'goc nhin phai la phep quay'
    return R


def rotate_mol(mol: Chem.Mol, R: np.ndarray) -> Chem.Mol:
    """Tra ve ban sao da quay toa do: Q = P @ R.T (cung quy uoc voi m2_align)."""
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, 'R phai co det=+1 (khong phan chieu!)'
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    Q = conf.GetPositions() @ R.T
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*Q[i]))
    return m

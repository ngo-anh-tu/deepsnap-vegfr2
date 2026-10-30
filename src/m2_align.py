"""
M2 - Can chinh phan tu vao he truc quan tinh chinh, TAT DINH va GIU CHIRALITY.

Hai dieu kien bat buoc:
  1. det(R) = +1  -> phep QUAY, khong phai phan chieu. Phan chieu se doi R <-> S.
  2. Quy uoc dau TAT DINH -> cung mot chat luon cho cung mot anh giua cac lan chay.

Quy tac dau (dieu ma ban setup truoc con bo trong):
  - Truc 1 va 2: chon dau sao cho mo men bac ba (co trong so Z) duong.
  - Neu |mo men bac ba| qua nho (phan tu doi xung) -> thu lan luot cac tieu chi du phong.
  - Truc 3 = cross(truc1, truc2)  -> tu dong dam bao det = +1.

[DO] tren tap VEGFR2: cap truc trong 2% cua nhau = 0,08% chat; trong 10% = 0,92%.
Cac chat do duoc danh co `axes_degenerate` de loai khoi phan tich can huong on dinh.
"""
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

TOL_MOMENT = 1e-6     # nguong coi mo men la "bang 0"
TOL_DEGEN  = 0.02     # hai tri rieng trong 2% cua nhau -> coi la suy bien


def _pick_sign(proj, w):
    """Tra ve +1/-1 tat dinh cho mot truc. Thu lan luot 4 tieu chi."""
    for val in (
        float(np.sum(w * proj**3)),      # 1. mo men bac ba co trong so Z
        float(np.sum(w * proj)),         # 2. mo men bac nhat co trong so Z
        float(np.sum(proj**3)),          # 3. mo men bac ba khong trong so
        float(proj[np.argmax(np.abs(proj))]),   # 4. nguyen tu xa nhat
    ):
        if abs(val) > TOL_MOMENT:
            return 1.0 if val > 0 else -1.0
    return 1.0                            # hoan toan doi xung: giu nguyen, van tat dinh


def align(mol: Chem.Mol, in_place: bool = False):
    """Can chinh mol (da co conformer). Tra ve (mol_da_can, info)."""
    m = mol if in_place else Chem.Mol(mol)
    conf = m.GetConformer()
    P = conf.GetPositions()
    heavy = np.array([a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1])
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=float)

    c = P[heavy].mean(axis=0)
    X = P[heavy] - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    a1, a2 = Vt[0].copy(), Vt[1].copy()
    w = Z[heavy]
    a1 *= _pick_sign(X @ a1, w)
    a2 *= _pick_sign(X @ a2, w)
    a2 -= np.dot(a2, a1) * a1                      # giu truc giao sau khi doi dau
    n2 = np.linalg.norm(a2)
    a2 = a2 / n2 if n2 > 1e-9 else Vt[1]
    a3 = np.cross(a1, a2)                          # <- dam bao det = +1

    R = np.stack([a1, a2, a3])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, 'can chinh lam mat chirality!'

    Q = (P - c) @ R.T
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*Q[i]))

    ext = S / np.sqrt(max(len(heavy) - 1, 1))
    degen = bool((ext[0] - ext[1]) / ext[0] < TOL_DEGEN or
                 (ext[1] - ext[2]) / ext[1] < TOL_DEGEN)
    info = dict(ext1=float(ext[0]), ext2=float(ext[1]), ext3=float(ext[2]),
                flat_ratio=float(ext[2] / ext[0]),     # <- cot cho phan tang §8.3
                axes_degenerate=degen, det=float(np.linalg.det(R)))
    return m, info
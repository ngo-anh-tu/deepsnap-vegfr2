"""
M1 - SMILES -> conformer 3D (ETKDGv3 + MMFF94).

Chinh sach that bai: thu lai 3 seed roi MOI bo. Chat bi bo duoc GHI LOG, khong bo im lang,
vi neu no nam trong test thi N doi -> mat kha nang so sanh voi baseline.
"""
from dataclasses import dataclass
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

EMBED_SEEDS = (0xf00d, 0xbeef, 0x1234)
MMFF_ITERS  = 800


@dataclass
class ConfResult:
    smiles: str
    mol: Chem.Mol | None
    seed_used: int | None
    mmff_energy: float | None
    ok: bool
    reason: str = ''


def embed(smiles: str, seeds=EMBED_SEEDS, add_hs: bool = True) -> ConfResult:
    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        return ConfResult(smiles, None, None, None, False, 'parse_failed')
    m = Chem.AddHs(m0) if add_hs else m0
    for sd in seeds:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = sd
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(m, ps) >= 0:
            e = None
            try:
                AllChem.MMFFOptimizeMolecule(m, maxIters=MMFF_ITERS)
                props = AllChem.MMFFGetMoleculeProperties(m)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(m, props)
                    if ff is not None:
                        e = float(ff.CalcEnergy())
            except Exception:
                pass
            return ConfResult(smiles, m, sd, e, True)
    return ConfResult(smiles, None, None, None, False, 'embed_failed_all_seeds')
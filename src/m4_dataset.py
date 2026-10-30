"""
M4 - Dataset doc PNG tu index.parquet. Moi item = MOT PHAN TU gom n_view anh.

Quy uoc:
  - x: [V,3,H,W] float32 trong [0,1] (chuan hoa ImageNet nam trong model, M5).
  - Augment/TTA chi la phep quay nho + tinh tien, det > 0. KHONG BAO GIO lat guong (lat = doi R <-> S).
  - split 'test' bi khoa: chi 08_final_test.py moi duoc mo (allow_test=True). 03_train.py khong bao gio.
"""
import math
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def affine_theta(rot_deg: float, dx_px: float, dy_px: float, size) -> torch.Tensor:
    """Ma tran 2x3 cho F.affine_grid (toa do chuan hoa). size: so hoac (W, H).
    Phep quay tinh trong khong gian PIXEL roi doi sang toa do chuan hoa, nen anh khong vuong
    khong bi meo; det = +1."""
    from src.config import wh
    W, H = wh(size)
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, -s * H / W, 2.0 * dx_px / W],
                         [s * W / H, c, 2.0 * dy_px / H]], dtype=torch.float32)


def apply_affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """x: [N,3,H,W] nen trang; theta: [2,3] hoac [N,2,3]. Vung trong sau bien doi van la TRANG."""
    n = x.shape[0]
    th = theta.expand(n, 2, 3) if theta.dim() == 2 else theta
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    return 1.0 - F.grid_sample(1.0 - x, grid, mode='bilinear', padding_mode='zeros',
                               align_corners=False)


def tta_thetas(cfg: dict) -> list:
    size = cfg['render']['image_px']
    return [affine_theta(t.get('rot_deg', 0.0), t.get('dx', 0.0), t.get('dy', 0.0), size)
            for t in cfg['tta']['transforms']]


def sample_train_theta(cfg: dict) -> torch.Tensor:
    a, size = cfg['augment'], cfg['render']['image_px']
    u = lambda lim: (torch.rand(()).item() * 2 - 1) * lim
    return affine_theta(u(a['rot_deg']), u(a['shift_px']), u(a['shift_px']), size)


class RenderDataset(Dataset):
    def __init__(self, render_dir, split: str, cfg: dict, train: bool = False,
                 allow_test: bool = False, use_cache: bool = False):
        if split == 'test' and not allow_test:
            raise PermissionError("split 'test' bi khoa (K.7): chi 08_final_test.py duoc mo")
        self.dir, self.cfg, self.train = render_dir, cfg, train
        ix = pd.read_parquet(render_dir / 'index.parquet')
        ix['_row'] = np.arange(len(ix))              # hang trong cache.npy (build_cache dung dung thu tu nay)
        ix = ix[ix['split'] == split].sort_values(['mol_idx', 'view_idx'])
        lim = cfg['data'].get('limit_mols')
        if lim:
            ix = ix[ix['mol_idx'].isin(ix['mol_idx'].unique()[:lim])]
        self.index = ix.reset_index(drop=True)
        g = self.index.groupby('mol_idx', sort=True)
        self.mol_idx = np.array(list(g.groups.keys()))
        self.rows = [grp.index.to_numpy() for _, grp in g]
        self.labels = np.array([self.index.at[r[0], 'label'] for r in self.rows], dtype=np.float32)
        self.smiles = [self.index.at[r[0], 'smiles'] for r in self.rows]
        self.cache = None
        if use_cache and (render_dir / 'cache.npy').exists():
            self.cache = np.load(render_dir / 'cache.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def _load(self, i):
        if self.cache is not None:
            return np.asarray(self.cache[int(self.index.at[i, '_row'])])
        return np.asarray(Image.open(self.dir / self.index.at[i, 'png_path']).convert('RGB'))

    def __getitem__(self, k):
        imgs = np.stack([self._load(i) for i in self.rows[k]])           # [V,H,W,3] uint8
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0     # [V,3,H,W]
        if self.train:
            x = apply_affine(x, sample_train_theta(self.cfg))              # cung 1 phep cho moi view
        return x, torch.tensor(self.labels[k]), int(self.mol_idx[k])

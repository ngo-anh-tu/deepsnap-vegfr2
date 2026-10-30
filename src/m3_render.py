"""
M3 - Render conformer 3D thanh anh, THANG TUYET DOI (A/pixel co dinh).

Diem chot: KHONG BAO GIO goi cmd.zoom() / cmd.orient() / cmd.center().
Thang duoc dat bang camera z, tu hieu chuan mot lan khi khoi tao.

Quan he da do thuc nghiem: px_per_A = k / |camera_z|  (orthoscopic, k ~ 974 o fov=20).
k phu thuoc field_of_view va phien ban PyMOL, nen module TU DO k thay vi hardcode.
"""
import os, numpy as np
from PIL import Image
from rdkit import Chem
import pymol2

VDW_SCALE   = 0.30      # ball-and-stick
STICK_R     = 0.12
FRAME_A     = 32.0      # khung tuyet doi, A  (max do duoc: 31.1 A)
IMG_PX      = 256
PX_PER_A    = IMG_PX / FRAME_A    # = 8.0
CARBON      = 'gray50'  # co dinh: mau C tu dong cua PyMOL doi theo tung object -> confound
SUPERSAMPLE = 4         # ve o IMG_PX*4 roi thu ve IMG_PX bang LANCZOS (khu rang cua)


class Renderer:
    def __init__(self, img_px=IMG_PX, frame_a=FRAME_A, bg='white', supersample=SUPERSAMPLE,
                 show_h=True):
        """img_px, frame_a: so (khung vuong) hoac (rong, cao). px/A phai nhu nhau theo hai chieu."""
        from src.config import wh
        self.img_wh, frame_wh = tuple(int(n) for n in wh(img_px)), wh(frame_a)
        self.px_per_a = self.img_wh[1] / frame_wh[1]
        if abs(self.img_wh[0] / frame_wh[0] - self.px_per_a) > 1e-6:
            raise ValueError('px/A theo chieu rong %.4f != chieu cao %.4f'
                             % (self.img_wh[0] / frame_wh[0], self.px_per_a))
        self.frame_a = max(frame_wh)     # do day lop cat (slab) truoc/sau camera
        self.ss, self.show_h = int(supersample), show_h
        self._p = pymol2.PyMOL(); self._p.start()
        c = self._p.cmd
        c.set('orthoscopic', 1)          # BAT BUOC: thang tuyen tinh, khong phoi canh
        c.set('sphere_scale', VDW_SCALE)
        c.set('stick_radius', STICK_R)
        c.set('antialias', 2)
        c.set('ray_trace_mode', 0)
        c.set('specular', 0)             # tat highlight -> bot nhieu cho kenh phu sau nay
        c.set('depth_cue', 0)            # tat fog: fog ma hoa do sau thanh do sang -> confound
        c.set('ray_shadows', 0)
        c.set('ray_opaque_background', 1)
        c.bg_color(bg)
        self.cmd = c
        self._z = self._calibrate()

    def _png(self, path):
        w, h = self.img_wh
        self.cmd.png(path, width=w * self.ss, height=h * self.ss, ray=0)
        if self.ss > 1:
            Image.open(path).resize((w, h), Image.LANCZOS).save(path)

    def _load(self, mol, name):
        c = self.cmd
        c.delete('all')
        c.read_molstr(Chem.MolToMolBlock(mol), name)
        c.hide('everything'); c.show('spheres'); c.show('sticks')
        c.color(CARBON, '%s and elem C' % name)
        if not self.show_h:
            c.hide('everything', '%s and elem H' % name)

    # ---------- hieu chuan ----------
    def _probe_extent_px(self, path):
        a = np.array(Image.open(path).convert('L'))
        nz = np.argwhere(a < 250)
        if len(nz) == 0: return 0.0
        return float(max(np.ptp(nz[:,0]), np.ptp(nz[:,1])) + 1)

    def _calibrate(self, tmp=None):
        """Hieu chuan bang benzene: duong kinh thuc tinh tu toa do + ban kinh ve.
        Quan he px/A = k/|z| la chinh xac (orthoscopic), nen MOT lan do la du."""
        import tempfile
        from rdkit.Chem import AllChem
        tmp = tmp or tempfile.mkdtemp()
        c = self.cmd
        m = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
        ps = AllChem.ETKDGv3(); ps.randomSeed = 0xf00d; ps.useRandomCoords = True
        if AllChem.EmbedMolecule(m, ps) < 0:
            raise RuntimeError('hieu chuan: embed benzene that bai')
        AllChem.MMFFOptimizeMolecule(m, maxIters=800)
        PT = Chem.GetPeriodicTable()
        P = m.GetConformer().GetPositions()
        rad = np.array([PT.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()]) * VDW_SCALE
        if not self.show_h:              # H bi an -> hop bao chi tinh nguyen tu nang
            keep = np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()])
            P, rad = P[keep], rad[keep]
        # base_view dung rotation = I, nen phep chieu la mat phang x-y.
        # _probe_extent_px do HOP BAO cua pixel, nen gia tri mong doi cung phai la hop bao
        # cua phep chieu do -- KHONG phai duong kinh 3D (duong kinh 3D la max moi huong).
        ext_a = float(max(
            (P[:, 0] + rad).max() - (P[:, 0] - rad).min(),
            (P[:, 1] + rad).max() - (P[:, 1] - rad).min()))

        self._load(m, 'cal')
        f = os.path.join(tmp, 'cal.png')

        def shoot(z):
            v = [1,0,0, 0,1,0, 0,0,1, 0.0,0.0,z, 0.0,0.0,0.0,
                 abs(z)-self.frame_a, abs(z)+self.frame_a, 20.0]
            c.set_view(v)
            self._png(f)
            return self._probe_extent_px(f) / ext_a

        z0 = -120.0
        got = shoot(z0)
        if got <= 0:
            raise RuntimeError('hieu chuan that bai: anh trong')
        k = abs(z0) * got
        z = -k / self.px_per_a
        chk = shoot(z)
        c.delete('all')
        self.calib = dict(k=k, z=z, ext_cal_a=ext_a,
                          px_per_a_target=self.px_per_a, px_per_a_measured=chk)
        return z

    def base_view(self):
        v = [1,0,0, 0,1,0, 0,0,1,
             0.0, 0.0, self._z,
             0.0, 0.0, 0.0,
             abs(self._z)-self.frame_a, abs(self._z)+self.frame_a, 20.0]
        return v

    # ---------- render ----------
    def render(self, mol, rot=np.eye(3), name='m', out=None):
        """mol: RDKit Mol DA can chinh (M2). rot: phep quay 3x3 (det=+1) cho goc nhin."""
        assert abs(np.linalg.det(rot) - 1.0) < 1e-6, 'rot phai co det=+1 (khong phan chieu!)'
        self._load(mol, name)
        v = self.base_view()
        v[0:9] = list(np.asarray(rot, float).ravel())
        self.cmd.set_view(v)               # <-- khong zoom, khong orient
        self._png(out)
        return out

    def close(self):
        try: self._p.stop()
        except Exception: pass
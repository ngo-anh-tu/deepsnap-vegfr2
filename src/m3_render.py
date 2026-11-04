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


class Renderer:
    def __init__(self, img_px=IMG_PX, frame_a=FRAME_A, bg='white', supersample=1, carbon_color=None,
                 sphere_scale=VDW_SCALE, stick_radius=STICK_R, element_colors=None, hide_nonpolar_h=False):
        """supersample=s: ve o img_px*s roi thu ve img_px (LANCZOS) -> canh muot, thang KHONG doi.
        carbon_color: to C mot mau co dinh; None = auto_color cua PyMOL (doi mau theo thu tu nap, la nhieu).
        element_colors: {nguyen to: mau PyMOL} ghi de mau mac dinh (vd F palecyan gan nhu chim vao nen trang).
        hide_nonpolar_h: an H gan voi C (khong mang thong tin lien ket H), giu H phan cuc (N-H, O-H).
        Mac dinh cua moi tham so = hanh vi cu."""
        self.img_px, self.frame_a = img_px, frame_a
        self.px_per_a = img_px / frame_a
        self.ss, self.carbon_color = int(supersample), carbon_color
        self.sphere_scale, self.stick_radius = sphere_scale, stick_radius
        self.element_colors, self.hide_nonpolar_h = dict(element_colors or {}), hide_nonpolar_h
        self.render_px = img_px * self.ss
        self._p = pymol2.PyMOL(); self._p.start()
        c = self._p.cmd
        c.set('orthoscopic', 1)          # BAT BUOC: thang tuyen tinh, khong phoi canh
        c.set('sphere_scale', sphere_scale)
        c.set('stick_radius', stick_radius)
        c.set('antialias', 2)
        c.set('ray_trace_mode', 0)
        c.set('specular', 0)             # tat highlight -> bot nhieu cho kenh phu sau nay
        c.set('depth_cue', 0)            # tat fog: fog ma hoa do sau thanh do sang -> confound
        c.set('ray_shadows', 0)
        c.set('ray_opaque_background', 1)
        c.bg_color(bg)
        self.cmd = c
        self._z = self._calibrate()

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
        rad = np.array([PT.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()]) * self.sphere_scale
        # base_view dung rotation = I, nen phep chieu la mat phang x-y.
        # _probe_extent_px do HOP BAO cua pixel, nen gia tri mong doi cung phai la hop bao
        # cua phep chieu do -- KHONG phai duong kinh 3D (duong kinh 3D la max moi huong).
        ext_a = float(max(
            (P[:, 0] + rad).max() - (P[:, 0] - rad).min(),
            (P[:, 1] + rad).max() - (P[:, 1] - rad).min()))

        c.delete('all')
        c.read_molstr(Chem.MolToMolBlock(m), 'cal')
        c.hide('everything'); c.show('spheres'); c.show('sticks')
        f = os.path.join(tmp, 'cal.png')

        def shoot(z):
            v = [1,0,0, 0,1,0, 0,0,1, 0.0,0.0,z, 0.0,0.0,0.0,
                 abs(z)-self.frame_a, abs(z)+self.frame_a, 20.0]
            c.set_view(v)
            c.png(f, width=self.render_px, height=self.render_px, ray=0)
            return self._probe_extent_px(f) / ext_a / self.ss     # do o render_px, quy ve px cua anh CUOI

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
        c = self.cmd
        c.delete('all')
        c.read_molstr(Chem.MolToMolBlock(mol), name)
        c.hide('everything'); c.show('spheres'); c.show('sticks')
        if self.carbon_color:
            c.color(self.carbon_color, 'elem C')
        for el, col in self.element_colors.items():
            c.color(col, 'elem %s' % el)
        if self.hide_nonpolar_h:
            c.hide('everything', 'hydro and neighbor elem C')
        v = self.base_view()
        v[0:9] = list(np.asarray(rot, float).ravel())
        c.set_view(v)                      # <-- khong zoom, khong orient
        c.png(out, width=self.render_px, height=self.render_px, ray=0)
        if self.ss > 1:
            Image.open(out).resize((self.img_px, self.img_px), Image.LANCZOS).save(out)
        return out

    def close(self):
        try: self._p.stop()
        except Exception: pass
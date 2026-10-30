"""
Xem thu: render cac goc nhin cua MOT phan tu, khong can conformers.sdf, khong train.
Dung dung chuoi cua 02_render.py: embed (M1) -> can chinh (M2) -> quay theo view -> render (M3).

Vi du:  python scripts/10_preview_views.py                    # phan tu dau tien, 3 goc V3
        python scripts/10_preview_views.py --row 5 --out reports/preview
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import RDLogger
from src.config import load_config, abs_path
from src.m1_conformer import embed
from src.m2_align import align
from src.views import view_matrix, rotate_mol
RDLogger.DisableLog('rdApp.*')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml', 'configs/render_v3.yaml'])
    ap.add_argument('--row', type=int, default=0, help='hang trong split_all.csv (0 = phan tu dau tien)')
    ap.add_argument('--out', default='reports/preview')
    a = ap.parse_args()
    cfg = load_config(*a.config); r = cfg['render']

    s = pd.read_csv(abs_path(cfg['paths']['split_csv'])).iloc[a.row]
    idx, smi = int(s['orig_idx']), s[cfg['data']['smiles_col']]
    print('mol %d | label %.3f | %s\n%s' % (idx, s[cfg['data']['label_col']], s['split'], smi))

    res = embed(smi)
    if not res.ok:
        raise SystemExit('embed that bai: %s' % res.reason)
    m = align(res.mol)[0] if r['align'] else res.mol

    from src.m3_render import Renderer
    R = Renderer(img_px=r['image_px'], frame_a=r['frame_A'], bg=r['background'],
                 supersample=r['supersample'], show_h=r['hydrogens'])
    out = abs_path(a.out); out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for vi, v in enumerate(r['views']):
        p = out / ('mol%06d_v%02d_%s.png' % (idx, vi, v['name']))
        R.render(rotate_mol(m, view_matrix(v)), out=str(p))
        im = Image.open(p).convert('RGB')
        ImageDraw.Draw(im).text((6, 4), v['name'], fill=(0, 0, 0))
        tiles.append(im); print('  ', p)
    R.close()

    sheet = Image.new('RGB', (sum(t.width for t in tiles), tiles[0].height), 'white')
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * t.width, 0))
    sp = out / ('mol%06d_views.png' % idx)
    sheet.save(sp); print('ghep:', sp)


if __name__ == '__main__':
    main()

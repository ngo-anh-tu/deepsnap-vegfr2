"""
Gom PNG thanh tam de luot bang mat: 100 chat / tam (10 x 10), ghi vao data/renders/<name>/qa/.
Moi o ghi mol_idx va pIC50, tra nguoc smiles qua index.parquet.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from src.config import load_config, render_dir

PER_SHEET, COLS, THUMB = 100, 10, 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml', 'configs/render_v1.yaml'])
    ap.add_argument('--view', type=int, default=0, help='view_idx can xem')
    ap.add_argument('--split', default=None, help='chi lay train/val/test (mac dinh: tat ca)')
    ap.add_argument('--max-sheets', type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(*a.config)
    d = render_dir(cfg)
    ix = pd.read_parquet(d / 'index.parquet')
    ix = ix[ix['view_idx'] == a.view]
    if a.split:
        ix = ix[ix['split'] == a.split]
    ix = ix.sort_values('mol_idx').reset_index(drop=True)
    n_sheets = (len(ix) + PER_SHEET - 1) // PER_SHEET
    if a.max_sheets:
        n_sheets = min(n_sheets, a.max_sheets)
    (d / 'qa').mkdir(exist_ok=True)

    for s in tqdm(range(n_sheets), desc='contact sheet'):
        chunk = ix.iloc[s * PER_SHEET:(s + 1) * PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new('RGB', (COLS * THUMB, rows * THUMB), 'white')
        dr = ImageDraw.Draw(sheet)
        for k, (_, r) in enumerate(chunk.iterrows()):
            im = Image.open(d / r['png_path']).convert('RGB')
            im.thumbnail((THUMB, THUMB), Image.LANCZOS)          # giu ti le khung (co the khong vuong)
            x, y = (k % COLS) * THUMB, (k // COLS) * THUMB
            sheet.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
            dr.rectangle([x, y, x + THUMB - 1, y + THUMB - 1], outline=(210, 210, 210))
            dr.text((x + 3, y + 2), '%d  %.2f' % (r['mol_idx'], r['label']), fill=(0, 0, 0))
        sheet.save(d / 'qa' / ('contact_sheet_%03d.jpg' % s), quality=85)
    print('viet %d tam -> %s' % (n_sheets, d / 'qa'))


if __name__ == '__main__':
    main()

"""
Buoc 2: conformer -> PNG (M2 can chinh + M3 render, thang tuyet doi). PNG la nguon chan ly (docs §12.1).

Ra: data/renders/<name>/{png/NNNN/molXXXXXX_vVV.png, index.parquet, MANIFEST.json, qa/scale_*.png}
    va cache.npy neu co --cache (tuy chon, gitignore; KHONG dung cho V27).
Chay lai se bo qua mol da du anh (--force de ve lai).
"""
import argparse, datetime, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger
from tqdm import tqdm
from src.config import load_config, abs_path, render_dir, config_sha256, git_sha, wh
from src.m1_conformer import embed
from src.m2_align import align
from src.views import view_matrix, rotate_mol
RDLogger.DisableLog('rdApp.*')

_R = None      # Renderer cua tung worker (PyMOL khong chia se qua process)
_CFG = None


def _png_rel(mol_idx, view_idx, shard):
    return 'png/%04d/mol%06d_v%02d.png' % (mol_idx // shard, mol_idx, view_idx)


def _init_worker(cfg):
    global _R, _CFG
    from src.m3_render import Renderer
    _CFG = cfg
    r = cfg['render']
    _R = Renderer(img_px=r['image_px'], frame_a=r['frame_A'], bg=r['background'],
                  supersample=r['supersample'], show_h=r['hydrogens'])


def prepare(mol, do_align):
    return align(mol)[0] if do_align else mol


def _render_one(task):
    mol_idx, block = task
    r = _CFG['render']; out_dir = Path(_CFG['_out_dir'])
    try:
        m = prepare(Chem.MolFromMolBlock(block, removeHs=False), r['align'])
        for vi, v in enumerate(r['views']):
            mv = rotate_mol(m, view_matrix(v))
            p = out_dir / _png_rel(mol_idx, vi, r['shard_size'])
            p.parent.mkdir(parents=True, exist_ok=True)
            _R.render(mv, out=str(p))
        return mol_idx, True, '', _R.calib
    except Exception as e:
        return mol_idx, False, repr(e), _R.calib


def read_sdf(path):
    sup = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for m in sup:
        if m is not None:
            yield int(m.GetProp('orig_idx')), Chem.MolToMolBlock(m)


def check_m3_constants(r):
    import src.m3_render as m3
    for key, have in (('sphere_scale', m3.VDW_SCALE), ('stick_radius', m3.STICK_R),
                      ('carbon_color', m3.CARBON)):
        if (r[key] != have) if isinstance(have, str) else (abs(r[key] - have) > 1e-9):
            raise SystemExit('config render.%s=%s lech voi src/m3_render.py=%s' % (key, r[key], have))


def scale_qa(cfg, out):
    """Anh benzene vs sorafenib canh nhau: phep kiem thang tuyet doi §4 (chong loi cmd.zoom())."""
    from src.m3_render import Renderer
    r = cfg['render']
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    R = Renderer(img_px=r['image_px'], frame_a=r['frame_A'], bg=r['background'],
                 supersample=r['supersample'], show_h=r['hydrogens'])
    mols = [('benzene', 'c1ccccc1'),
            ('sorafenib', 'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1')]
    tiles = []
    for nm, smi in mols:
        m = prepare(embed(smi).mol, True)
        mv = rotate_mol(m, view_matrix(r['views'][0]))
        tmp = str(Path(out).with_name('_tmp_%s.png' % nm))
        R.render(mv, out=tmp)
        im = Image.open(tmp).convert('RGB'); Path(tmp).unlink()
        d = ImageDraw.Draw(im); d.text((6, 4), nm, fill=(0, 0, 0))
        tiles.append(im)
    W = sum(t.width for t in tiles); sheet = Image.new('RGB', (W, tiles[0].height), 'white')
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0)); x += t.width
    ImageDraw.Draw(sheet).text((6, sheet.height - 14),
        'px/A muc tieu=%.3f  do lai=%.3f  (chat to phai CHIEM NHIEU anh hon, khong fit-to-frame)'
        % (R.calib['px_per_a_target'], R.calib['px_per_a_measured']), fill=(0, 0, 0))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    R.close()


def build_cache(out_dir, index, r):
    n, (w, h) = len(index), wh(r['image_px'])
    mm = np.lib.format.open_memmap(out_dir / 'cache.npy', mode='w+', dtype=np.uint8, shape=(n, h, w, 3))
    for i, p in enumerate(tqdm(index['png_path'], desc='cache.npy')):
        mm[i] = np.asarray(Image.open(out_dir / p).convert('RGB'))
    mm.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['configs/base.yaml', 'configs/render_v1.yaml'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--cache', action='store_true', help='dung cache.npy (memmap) tu png/')
    a = ap.parse_args()
    cfg = load_config(*a.config); r = cfg['render']
    check_m3_constants(r)
    out_dir = render_dir(cfg); out_dir.mkdir(parents=True, exist_ok=True)
    cfg['_out_dir'] = str(out_dir)

    split = pd.read_csv(abs_path(cfg['paths']['split_csv']))
    if a.limit or cfg['data']['limit_mols']:
        split = split.head(a.limit or cfg['data']['limit_mols'])
    want = set(split['orig_idx'])
    blocks = {i: b for i, b in read_sdf(abs_path(cfg['paths']['interim_dir']) / 'conformers.sdf') if i in want}
    n_embed_missing = len(want) - len(blocks)
    todo = []
    for idx, block in blocks.items():
        done = all((out_dir / _png_rel(idx, vi, r['shard_size'])).exists() for vi in range(len(r['views'])))
        if a.force or not done:
            todo.append((idx, block))

    calib, failed = None, []
    if todo:
        with Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for idx, ok, err, c in tqdm(pool.imap_unordered(_render_one, todo, chunksize=8),
                                        total=len(todo), desc='render'):
                calib = c
                if not ok:
                    failed.append((idx, err))
    else:
        print('khong co gi can ve (da du anh). --force de ve lai.')

    # index.parquet: chi liet ke anh DA CO tren dia
    rows = []
    for _, s in split.iterrows():
        for vi in range(len(r['views'])):
            rel = _png_rel(int(s['orig_idx']), vi, r['shard_size'])
            if (out_dir / rel).exists():
                rows.append(dict(mol_idx=int(s['orig_idx']), view_idx=vi, png_path=rel,
                                 smiles=s[cfg['data']['smiles_col']], label=float(s[cfg['data']['label_col']]),
                                 split=s['split']))
    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / 'index.parquet', index=False)

    scale_qa(cfg, out_dir / 'qa' / 'scale_benzene_vs_sorafenib.png')
    if calib is None:       # khong ve gi moi -> do lai thang de ghi MANIFEST
        from src.m3_render import Renderer
        R = Renderer(img_px=r['image_px'], frame_a=r['frame_A'], bg=r['background'],
                     supersample=r['supersample'], show_h=r['hydrogens']); calib = R.calib; R.close()
    if a.cache:
        build_cache(out_dir, index, r)

    import json, pymol2
    p = pymol2.PyMOL(); p.start(); pv = p.cmd.get_version(); p.stop()
    n_mol = index['mol_idx'].nunique()
    manifest = dict(
        git_sha=git_sha(), pymol_version=pv if isinstance(pv, str) else str(pv[0]),
        k_measured=calib['k'], px_per_A_actual=calib['px_per_a_measured'],
        frame_A=r['frame_A'], image_px=r['image_px'], supersample=r['supersample'],
        carbon_color=r['carbon_color'], hydrogens=r['hydrogens'], n_view=len(r['views']), view_name=r['view_name'],
        views=[v['name'] for v in r['views']], align=r['align'],
        config_sha256=config_sha256({k: v for k, v in cfg.items() if not k.startswith('_')}),
        n_molecules=int(n_mol), n_images=int(len(index)),
        n_failed=int(len(want) - n_mol), n_failed_embed=int(n_embed_missing), n_failed_render=len(failed),
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    with open(out_dir / 'MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False); f.write('\n')
    print('\n%d phan tu, %d anh, that bai %d (embed %d, render %d)'
          % (n_mol, len(index), manifest['n_failed'], n_embed_missing, len(failed)))
    for idx, err in failed[:20]:
        print('  mol %d: %s' % (idx, err))


if __name__ == '__main__':
    main()

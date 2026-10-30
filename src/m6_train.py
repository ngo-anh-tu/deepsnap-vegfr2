"""
M6 - Huan luyen 1 seed. CHI doc train + val; KHONG BAO GIO mo split test (K.7, docs §12.6).

Early-stop va bao cao dung CUNG mot dai luong: val RMSE muc phan tu, median qua goc nhin,
co TTA (§12.8 #3). Loss tinh tren TUNG ANH; danh gia tren median.
Ra trong out_dir: train_log.csv, best.ckpt, predictions_val.csv, metrics.json.
"""
import json, math, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import render_dir, config_sha256, git_sha
from src.m4_dataset import RenderDataset
from src.m5_model import build_model
from src.m7_eval import predict, regression_metrics


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _worker_init(_):
    torch.set_num_threads(1)         # tranh tranh chap CPU giua cac worker (augment chay tren CPU)


def _loader(ds, cfg, shuffle):
    t = cfg['train']
    nw = t['num_workers']
    return DataLoader(ds, batch_size=t['batch_size'], shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0, worker_init_fn=_worker_init,
                      drop_last=shuffle and len(ds) > t['batch_size'])


def train_seed(cfg: dict, seed: int, out_dir) -> dict:
    t = cfg['train']
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dev = select_device()
    rd = render_dir(cfg)
    uc = cfg['data'].get('use_cache', False)
    ds_tr = RenderDataset(rd, 'train', cfg, train=True, use_cache=uc)
    ds_tr_eval = RenderDataset(rd, 'train', cfg, train=False, use_cache=uc)     # de do TRAIN RMSE (§9.1), khong augment
    ds_va = RenderDataset(rd, 'val', cfg, train=False, use_cache=uc)
    dl_tr, dl_trE, dl_va = _loader(ds_tr, cfg, True), _loader(ds_tr_eval, cfg, False), _loader(ds_va, cfg, False)

    y_mean, y_std = float(ds_tr.labels.mean()), float(ds_tr.labels.std())
    model = build_model(cfg, y_mean, y_std).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    steps_ep = max(len(dl_tr), 1)
    total, warm = t['epochs'] * steps_ep, t['warmup_epochs'] * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=t['amp'] and dev.type == 'cuda')
    use_tta = cfg['tta']['enabled'] and 'val' in cfg['tta']['apply_to']

    log, best, best_epoch, bad = [], float('inf'), -1, 0
    for ep in range(t['epochs']):
        model.train(); t0 = time.time(); run = []
        for x, y, _ in tqdm(dl_tr, desc='seed%d ep%d' % (seed, ep), leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, enabled=t['amp'] and dev.type == 'cuda'):
                pred = model(x).float()                                   # [B,V] tung anh
            loss = (((pred - y[:, None]) / model.y_std) ** 2).mean()      # MSE tren nhan chuan hoa
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'])
            scaler.step(opt); scaler.update(); sched.step()
            run.append(loss.item())
        va = predict(model, dl_va, dev, cfg, use_tta, t['amp'])
        m = regression_metrics(va['y_true'], va['y_pred'])
        log.append(dict(epoch=ep, train_loss=float(np.mean(run)), val_rmse=m['rmse'], val_mae=m['mae'],
                        val_r2=m['r2'], lr=opt.param_groups[0]['lr'], sec=time.time() - t0))
        pd.DataFrame(log).to_csv(out_dir / 'train_log.csv', index=False)
        if m['rmse'] < best:
            best, best_epoch, bad = m['rmse'], ep, 0
            torch.save(dict(model=model.state_dict(), epoch=ep, y_mean=y_mean, y_std=y_std,
                            seed=seed), out_dir / 'best.ckpt')
        else:
            bad += 1
            if bad >= t['patience']:
                break

    ck = torch.load(out_dir / 'best.ckpt', map_location=dev)
    model.load_state_dict(ck['model'])
    va = predict(model, dl_va, dev, cfg, use_tta, t['amp'])
    va.insert(1, 'smiles', ds_va.smiles)
    va.to_csv(out_dir / 'predictions_val.csv', index=False)
    tr = predict(model, dl_trE, dev, cfg, use_tta, t['amp'])
    metrics = dict(seed=seed, git_sha=git_sha(), config_sha256=config_sha256(cfg),
                   split='val', n_molecules=int(len(ds_va)),            # SO PHAN TU, khong phai so anh
                   n_images=int(len(ds_va.index)), best_epoch=best_epoch,
                   train_rmse=regression_metrics(tr['y_true'], tr['y_pred'])['rmse'],
                   **regression_metrics(va['y_true'], va['y_pred']))
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2); f.write('\n')
    return metrics

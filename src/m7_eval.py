"""
M7 - Du bao, chi so, bootstrap, gom seed, nhat ky test.

CANH BAO: moi khoang tin cay phai bootstrap theo PHAN TU (n_molecules), KHONG theo anh.
Bootstrap theo anh se hep sai sqrt(n_view) lan. Cac ham o day nhan mang MUC PHAN TU.
"""
import datetime, json
import numpy as np, pandas as pd, torch
from scipy import stats
from src.m4_dataset import apply_affine, tta_thetas
from src.m5_model import aggregate


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                r2=1.0 - ss_res / ss_tot,
                pearson=float(stats.pearsonr(y, p)[0]), spearman=float(stats.spearmanr(y, p)[0]))


def classification_metrics(y, p, thr: float) -> dict:
    """Nhi phan hoa pIC50 >= thr (active). AUC dung thang y_pred lam diem, khong can nguong thu hai.
    Lop lech (~77% active o thr=6) -> accuracy de danh lua, doc kem balanced_acc / MCC."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    yt, yp = y >= thr, p >= thr
    t = ('%g' % thr).replace('.', '_')
    return {'acc@%s' % t: float(accuracy_score(yt, yp)),
            'bal_acc@%s' % t: float(balanced_accuracy_score(yt, yp)),
            'mcc@%s' % t: float(matthews_corrcoef(yt, yp)),
            'auc@%s' % t: float(roc_auc_score(yt, p)),
            'active_frac@%s' % t: float(yt.mean())}


@torch.no_grad()
def predict(model, loader, device, cfg: dict, use_tta: bool, amp: bool = False) -> pd.DataFrame:
    """Tra ve DataFrame [mol_idx, y_true, y_pred] muc phan tu.
    TTA: trung binh cac phep (quay nho/tinh tien) tren tung anh, roi median qua cac goc nhin."""
    model.eval()
    thetas = tta_thetas(cfg) if use_tta else [None]
    rows = []
    for x, y, idx in loader:
        x = x.to(device)
        b, v = x.shape[:2]
        xf = x.flatten(0, 1)
        acc = 0.0
        for th in thetas:
            xt = xf if th is None else apply_affine(xf, th.to(device))
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                acc = acc + model.predict_images(xt).float()
        pred = aggregate((acc / len(thetas)).view(b, v))
        rows.append(pd.DataFrame(dict(mol_idx=idx.numpy(), y_true=y.numpy(), y_pred=pred.cpu().numpy())))
    return pd.concat(rows, ignore_index=True)


def bootstrap_ci(y, p, metric='rmse', n=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    vals = np.array([regression_metrics(y[i], p[i])[metric]
                     for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(y, p_a, p_b, n=2000, ci=0.95, seed=0):
    """RMSE(a) - RMSE(b) tren CUNG tap phan tu. <0 nghia la a tot hon b."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = (np.asarray(v, float) for v in (y, p_a, p_b))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(np.sqrt(((p_a[i] - y[i]) ** 2).mean()) - np.sqrt(((p_b[i] - y[i]) ** 2).mean()))
    d = np.array(d)
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(d.mean()), float(lo), float(hi)


def aggregate_seeds(run_dir, keys=('rmse', 'mae', 'r2', 'pearson', 'spearman', 'train_rmse')) -> dict:
    """Gom seed*/metrics.json -> runs/<run>/metrics.json (mean +- std)."""
    seeds = [p for p in sorted(run_dir.glob('seed*/metrics.json')) if p.stat().st_size > 0]
    ms = [json.load(open(p)) for p in seeds]
    out = dict(run=run_dir.name, n_seeds=len(ms), seeds=[m['seed'] for m in ms],
               git_sha=ms[0].get('git_sha'), config_sha256=ms[0].get('config_sha256'),
               n_molecules=ms[0].get('n_molecules'))
    for k in keys:
        v = [m[k] for m in ms if k in m]
        if v:
            out[k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, values=v)
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2); f.write('\n')
    return out


# ---- nhat ky test (K.7): append-only, moi lan cham test 1 dong ----
LOG_HEAD = '| ngay | run | config_sha256 | ly do |\n|---|---|---|---|\n'


def test_already_logged(reports_dir, run: str) -> bool:
    p = reports_dir / 'test_set_log.md'
    return p.exists() and ('| %s |' % run) in p.read_text()


def append_test_log(reports_dir, run: str, config_sha256: str, reason: str):
    p = reports_dir / 'test_set_log.md'
    head = '# Nhat ky cham test set (append-only, K.7)\n\n' + LOG_HEAD if not p.exists() or p.stat().st_size == 0 else ''
    with open(p, 'a') as f:
        f.write('%s| %s | %s | %s | %s |\n' % (head, datetime.date.today().isoformat(), run,
                                             (config_sha256 or '-')[:12], reason))

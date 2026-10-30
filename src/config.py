"""
Doc + gop YAML, hash cau hinh, git sha. Moi script/module doc cau hinh qua day.
Duong dan trong `paths:` la tuong doi so voi goc repo.
"""
import copy, hashlib, json, subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths) -> dict:
    cfg = {}
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else ROOT / p
        with open(p) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def wh(v) -> tuple:
    """image_px / frame_A trong config: so (khung vuong) hoac [rong, cao]."""
    return (v, v) if isinstance(v, (int, float)) else (v[0], v[1])


def abs_path(rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def git_sha():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def render_dir(cfg: dict) -> Path:
    return abs_path(cfg['paths']['renders_dir']) / cfg['render']['name']


def run_name(cfg: dict) -> str:
    return '%s_%s' % (cfg['render']['name'], cfg['model']['name'])

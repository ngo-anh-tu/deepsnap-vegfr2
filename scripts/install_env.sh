#!/usr/bin/env bash
# Cai thu vien vao env conda DANG ACTIVE. Chay:  bash scripts/install_env.sh
# (chay bang bash de cac dong chu thich khong bi zsh hieu nham)
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_DEFAULT_ENV:-base}" = "base" ]; then
  echo "Hay 'conda activate <env>' (khong phai base) truoc khi chay." >&2
  exit 1
fi
echo "Cai vao env: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"

# pymol2 khong co tren PyPI -> bat buoc conda-forge. conda-forge osx-arm64 hien toi 3.1.0.
conda install -y -c conda-forge --override-channels \
  "python>=3.11" rdkit "pymol-open-source>=3.1" \
  "numpy>=2.0" "pandas>=2.2" scikit-learn scipy "pillow>=10.0" "pyarrow>=16.0" \
  pytorch "timm>=1.0" tqdm pyyaml pytest

echo "== kiem tra =="
python -c "import pymol2; p=pymol2.PyMOL(); p.start(); print('pymol', p.cmd.get_version()); p.stop()"
python -c "import rdkit, torch, timm, pandas, sklearn; print('rdkit', rdkit.__version__, '| torch', torch.__version__, '| timm', timm.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"

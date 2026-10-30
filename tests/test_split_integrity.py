"""split_all.csv: khong ro ri scaffold, khong trung, dung quy mo. Neu da render thi index khop split."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pytest
from src.config import ROOT

SPLIT = ROOT / 'data' / 'raw' / 'split_all.csv'
pytestmark = pytest.mark.skipif(not SPLIT.exists(), reason='thieu split_all.csv')


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(SPLIT)


def test_splits_and_sizes(df):
    assert set(df['split']) == {'train', 'val', 'test'}
    assert (df['split'] == 'test').sum() == 932      # §3; baseline bootstrap ghep cap dua vao so nay
    assert len(df) == 9306


def test_ids_unique_and_labels_finite(df):
    assert df['orig_idx'].is_unique
    assert df['label'].notna().all() and df['smiles'].notna().all()


def test_no_scaffold_crosses_splits(df):
    assert (df.groupby('scaffold')['split'].nunique() > 1).sum() == 0


def test_render_index_matches_split(df):
    idx = sorted((ROOT / 'data' / 'renders').glob('*/index.parquet'))
    idx = [p for p in idx if p.stat().st_size > 0]
    if not idx:
        pytest.skip('chua render')
    ref = df.set_index('orig_idx')['split']
    for p in idx:
        ix = pd.read_parquet(p)
        assert (ix['split'].values == ref.loc[ix['mol_idx']].values).all(), p
        assert ix.groupby('mol_idx')['split'].nunique().max() == 1

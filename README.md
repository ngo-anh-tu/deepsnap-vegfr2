# deepsnap-vegfr2

Dự đoán pIC50 của chất ức chế VEGFR2 từ **ảnh 3D** của phân tử (SMILES → conformer → căn chỉnh theo
trục chính → render PyMOL theo thang Å/pixel tuyệt đối → CNN ImageNet). Thiết kế và các quyết định nằm
trong [docs/deepsnap-vegfr2-setup.md](docs/deepsnap-vegfr2-setup.md).

## Dựng môi trường

`pymol2` không có trên PyPI nên phải dùng conda (§12.0); `requirements.txt` một mình không đủ.

```bash
conda env create -f environment.yml && conda activate deepsnap-vegfr2
make check          # pymol2 + audit split + tests
```

## Chạy

```bash
make conformers     # data/interim/{conformers.sdf, conformers_meta.parquet, geometry.parquet}
make render         # data/renders/v1/{png/, index.parquet, MANIFEST.json, qa/}
make sheet          # qa/contact_sheet_*.jpg để lướt bằng mắt
make baselines      # constant + ECFP4/GBM trên split_all.csv
make train          # 3 seed, CHỈ đọc train + val
make final-test RUN=runs/v1_r18 REASON="cấu hình đã chốt"   # test chạy đúng MỘT lần
```

Trước khi tin ảnh: mở `data/renders/v1/qa/scale_benzene_vs_sorafenib.png` — sorafenib phải chiếm nhiều
ảnh hơn benzene (thang tuyệt đối, không fit-to-frame).

## Kỷ luật test set (K.7)

`03_train.py` không load split `test`. Chỉ `scripts/08_final_test.py` mở nó, từ chối chạy lần hai trên
cùng một run, và ghi một dòng vào `reports/test_set_log.md`. Baseline cũng ghi vào nhật ký đó.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `configs/` | `base.yaml` + `render_*.yaml` + `model_*.yaml` (+ `ablation/`), gộp theo thứ tự |
| `src/` | M1 conformer · M2 căn chỉnh · M3 render · M4 dataset · M5 model · M6 train · M7 eval |
| `scripts/` | các bước đánh số theo thứ tự chạy |
| `baselines/` | hằng số, ECFP4+GBM — cùng schema `predictions_test.csv` |
| `data/` | `raw/` (chép vào) · `interim/` và `renders/` (sinh ra) |
| `runs/<run>/seed*/` | `train_log.csv`, `best.ckpt`, `predictions_val.csv`, `metrics.json` |

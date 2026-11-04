# §12. Khung repo, quy ước file, và ba quyết định còn lại

Phiên 2026-10-27, nối sau §11. Mục này chốt **cấu trúc thư mục, quy ước lưu ảnh, và ba tham số
chưa có tên trường** — tất cả những gì cần trước khi gõ dòng code module đầu tiên.

Ký hiệu nguồn vẫn như §1: **[ĐO]** · **[GỐC]** · **[VL]** · **[CHUẨN]** · **[SUY]**.

---

## 12.0 ⚠ Blocker phải kiểm TRƯỚC khi viết thêm bất cứ thứ gì

`m3_render.py` đã viết xong, nhưng **`pymol2` không có trên PyPI**. Nếu môi trường không dựng được
nó thì toàn bộ M3 phải viết lại bằng renderer khác, và `configs/render_v1.yaml` (chứa `k`,
`set_view`, `sphere_scale`) cũng thành vô nghĩa theo.

```bash
conda install -c conda-forge pymol-open-source
python -c "import pymol2; p=pymol2.PyMOL(); p.start(); print(p.cmd.get_version())"
```

Chạy được → viết `environment.yml` rồi đi tiếp. Không chạy được → dừng, tính lại M3.

`requirements.txt` một mình **không đủ** để dựng lại dự án này.

---

## 12.1 Quy ước render: PNG là nguồn chân lý, không phải `.npy`  **[SUY]**

Quyết định: **ảnh PNG là artifact thật, xem được bằng mắt, và là thứ mọi bước sau đọc vào.**
Mảng `.npy` hạ xuống thành cache tùy chọn.

Ba lý do:

1. **Dung lượng.** `images.npy` cho V27 = 251.262 × 256 × 256 × 3 uint8 = **45 GiB**. PNG cùng bộ
   ≈ 10 GB. V1 thì 370 MB so với 1,7 GiB.
2. **Không mở xem được.** Mục đích của nhánh ảnh là *nhìn thấy* phân tử; một mảng nhị phân 45 GiB
   không phục vụ việc đó.
3. **Không phải nút cổ chai.** Decode PNG 256×256 ≈ 1 ms; 8 worker thừa sức nuôi ResNet-18 trên 3090.

Cấu trúc một thư mục render:

```
data/renders/v1/
    png/
        0000/  mol000000_v00.png … mol000999_v00.png    ← shard theo mol_idx // 1000
        0001/  …
    index.parquet     ← mol_idx, view_idx, png_path, smiles, label, split
    MANIFEST.json     ← COMMIT
    cache.npy         ← tùy chọn · gitignore · dựng lại được từ png/
    qa/
        scale_benzene_vs_sorafenib.png    ← phép kiểm thang tuyệt đối ở §4
        contact_sheet_000.jpg …           ← 100 chất/tấm, để lướt bằng mắt
```

- **Shard `png/`**: 251k file trong một thư mục làm `ls`, Finder và rsync đều lết. 1.000 file/thư
  mục thì mở xem bình thường.
- **Tên file mang `mol_idx` + `view_idx`**, tra ngược qua `index.parquet` (có sẵn `smiles` và
  `label` để biết đang nhìn chất nào, pIC50 bao nhiêu).
- **`qa/` là bắt buộc, không phải trang trí.** Ảnh benzene vs sorafenib chính là phép kiểm 5 phút ở
  §4 chống lỗi `cmd.zoom()` — nó *phải* là ảnh để nhìn. Contact sheet là cách duy nhất thực tế để
  soi 9.306 chất.
- Với **V27**: giữ `png/` đầy đủ, **bỏ hẳn `cache.npy`**. Nếu vẫn muốn cache, ghi bằng
  `np.lib.format.open_memmap` rồi điền từng dòng — đừng `np.stack` (45 GiB vào RAM).

Dung lượng PNG đầy đủ: **V1 ≈ 370 MB · V3 ≈ 1,1 GB · V27 ≈ 10 GB.** Chịu được cả ba.

### Nội dung tối thiểu của `MANIFEST.json`

Đây là thứ làm phép kiểm toán ở §4 chạy được, nên không được thiếu trường nào:

| Trường | Vì sao |
|---|---|
| `git_sha` | Bộ ảnh sinh từ code nào |
| `pymol_version` | `k` phụ thuộc phiên bản |
| `k_measured` | Giá trị `_calibrate` tự đo được (không hardcode, §4) |
| `px_per_A_actual` | Thực đạt — kỳ vọng ≈ 8,087 |
| `frame_A`, `image_px`, `n_view`, `view_name` | Cấu hình render |
| `config_sha256` | Hash của YAML đã merge |
| `n_molecules`, `n_images`, `n_failed` | Đếm được |
| `timestamp` | |

---

## 12.2 Cây thư mục chốt

| Folder | File | Trạng thái |
|---|---|---|
| `deepsnap-vegfr2/` | `README.md` · `Makefile` · `.gitignore` | cần viết |
| | `requirements.txt` | ✅ đã có |
| | **`environment.yml`** | **cần viết — pymol không cài bằng pip (§12.0)** |
| `configs/` | `base.yaml` · `render_v1.yaml` · `model_r18.yaml` | cần viết — **3 file này trước** |
| | `render_v3.yaml` · `render_v27.yaml` · `model_effb0.yaml` | sau bước 1 |
| | ~~`model_r50.yaml`~~ | **bỏ** — §9.1: chỉ dựng nếu underfit |
| `configs/ablation/` | `res_128.yaml` · `scratch.yaml` · `no_align.yaml` | sau bước 1 |
| `data/raw/` | `VEGFR2_dataset_raw.xlsx` · `split_all.csv` | chép vào |
| | `kinase_chembl/` | rỗng, cho §8.4 ★ |
| `data/interim/` | `conformers.sdf` · `conformers_meta.parquet` | ⚙ sinh |
| | `geometry.parquet` | ⚙ sinh — **COMMIT** · xem §12.4 |
| `data/renders/v1/` | `png/` · `index.parquet` · `MANIFEST.json` · `qa/` | ⚙ sinh — commit `MANIFEST.json` + `index.parquet` |
| | `cache.npy` | ⚙ tùy chọn · gitignore |
| `src/` | `__init__.py` | cần viết |
| | `m1_conformer.py` · `m2_align.py` · `m3_render.py` | ✅ đã có |
| | `m4_dataset.py` · `m5_model.py` · `m6_train.py` · `m7_eval.py` | cần viết — **cụm chính** |
| | `m8_report.py` | sau khi có số đầu tiên |
| `baselines/` | `constant.py` · `ecfp_gbm.py` | cần viết — chạy **trên `split_all.csv`** |
| `tests/` | `test_scale_absolute.py` · `test_frame_fits.py` | ✅ đã có |
| | **`test_align_deterministic.py`** | **thêm mới — §12.3** |
| | `test_no_mirror.py` · `test_split_integrity.py` | cần viết |
| | ~~`test_pool_level.py`~~ | **bỏ** — conformer pool để vòng sau (§1 dòng 21) |
| `scripts/` | **`00_audit_split.py`** | **thêm — §12.5** |
| | `00_check_pipeline.sh` | cần viết |
| | `01_build_conformers.py` | cần viết — **tiếp theo** |
| | `02_render.py` | ✅ đã có |
| | `03_train.py` | cần viết — **không đụng test** (§12.6) |
| | **`08_final_test.py`** | **thêm — §12.6** |
| | **`09_contact_sheet.py`** | **thêm — gom PNG thành tấm lướt** |
| | `04_ablate_views.py` · `05_ablate_resolution.py` · `06_stratify_flatness.py` · `07_pretrain_kinase.py` | sau bước 1 |
| `runs/v1_r18/` | `config.yaml` · `metrics.json` · `seed0/ seed1/ seed2/` | ⚙ sinh — §12.6 |
| `reports/` | `split_audit.md` · `results_v1.md` · `bootstrap_vs_baseline.md` | cần viết |
| | `test_set_log.md` | append-only, mỗi lần chạm test 1 dòng |
| | `ablation_views.md` · `ablation_resolution.md` · `stratified_flatness.md` · `chirality_probe.md` | sau |
| `docs/` | `deepsnap-vegfr2-setup.md` | ✅ đã có |
| | **`deepsnap-data-audit.md` · `chemprop-baseline-data-check.md`** | **copy vào repo** |
| | `METHODS.md` | cần viết |

**Về hai doc phải copy vào `docs/`:** mọi số **[ĐO]** ở §2 và mọi tham chiếu K.* trong tài liệu này
hiện trỏ ra ngoài repo. Người mở repo ra không tra được chúng.

**Thống nhất tên:** §4 ghi `code/m3_render.py::_calibrate` — phải sửa thành `src/`.

---

## 12.3 Test thứ ba: tính tất định của M2  **[SUY]**

§4 đã tự cảnh báo rằng quy ước dấu theo mô men bậc ba **cần quy tắc phá thế** khi
|mô men bậc ba| ≈ 0, *"nếu không ảnh KHÔNG tái lập được giữa các lần"*. Đó là **lỗi im lặng thứ ba**,
ngang hàng với hai lỗi đã có test ở §4 (`cmd.zoom()` và early-stop sai mức).

`tests/test_align_deterministic.py` kiểm ba điều:

1. Chạy M2 hai lần trên cùng một chất → `coords_aligned` khớp **bit-level**.
2. Chất đối xứng cao, nơi mô men bậc ba ≈ 0: **benzene · adamantane · 4,4'-bipyridine** → vẫn tất định.
3. `det(Vt) = +1` trên mẫu ngẫu nhiên → xác nhận là phép quay, không phản chiếu (không phá chirality).

[ĐO] trên tập này: cặp trục trong 2% của nhau = 0,08% chất; trong 10% = 0,92%. Nhỏ, nhưng phải có
quy tắc tất định.

---

## 12.4 `geometry.parquet` — mục duy nhất có hạn sử dụng

Phải tính **ngay trong `01_build_conformers.py`**, cùng lần chạy sinh conformer:

| Cột | Vì sao phải có từ đầu |
|---|---|
| `flat_ratio` (= ext3/ext1) | Ngưỡng 0,25 ở §8.3 chỉ có giá trị nếu được chốt **trước** khi thấy kết quả train. Tính sau là mất phép đăng ký trước |
| `has_undefined_stereo` | 633 chất (6,78%, §2.6) phải bị loại khỏi mọi phân tích chirality — cờ này cũng phải có trước |
| `diameter_vdw`, `radius_gyration`, `ext1/ext2/ext3` | §2.4, dùng cho QA thang và phân tầng |
| `n_heavy`, `n_atoms`, `embed_seed`, `embed_ok` | Truy vết M1 |

Đây là lý do `geometry.parquet` nằm trong nhóm "⚙ script sinh" nhưng **vẫn phải commit**.

---

## 12.5 `split_all.csv` là file dẫn xuất — cần script audit

Chuỗi tạo ra nó (raw 12.572 → clean 9.341 → cap 96 → 9.306) hiện chỉ tồn tại trong
`chemprop-baseline-data-check.md`, ngoài repo. Người tái lập mở repo ra sẽ thấy một file split từ
trên trời rơi xuống.

`scripts/00_audit_split.py` sinh `reports/split_audit.md` — **chính bảng §3**, chạy được chứ không
chép tay: 0 scaffold xuyên split / 3.521 · lệch mean test−train = −0,063 · RMSE hằng số = 1,091 ·
quy ước `MurckoScaffoldSmiles(includeChirality=True)` khớp 100,0% · 35 chất bị loại so với 9.341 ·
10/75 cặp enantiomer bị tách, trong đó 3 cặp train↔test.

Chạy kèm trong `00_check_pipeline.sh`.

---

## 12.6 Kỷ luật test set phải nằm trong cấu trúc thư mục

K.7 ghi: *"Từ đây chỉ đọc val cho mọi thí nghiệm; test chạy một lần duy nhất với cấu hình đã chốt."*
Nếu `predictions_test.csv` nằm mặc định trong mỗi `runs/<run>/` thì **mỗi lần train là một lần chạm
test**. Theo K.9, kỷ luật đánh giá là phần có giá trị nhất của dự án — nên để nó nằm trong cấu trúc,
không chỉ trong trí nhớ.

**Quy tắc:**

- `03_train.py` **không** sinh `predictions_test.csv`, không load split `test`.
- `08_final_test.py` nhận một run dir đã chốt, chạy một lần, và **append một dòng** vào
  `reports/test_set_log.md`: ngày · run · config hash · lý do chạy.

**Thư mục run chịu được 3–5 seed** (§5 chốt 3 cho thí nghiệm, 5 cho kết luận):

```
runs/v1_r18/
    config.yaml              ← YAML đã merge
    metrics.json             ← mean ± std qua các seed
    seed0/ seed1/ seed2/
        train_log.csv
        best.ckpt
        predictions_val.csv
        metrics.json
```

Mỗi `metrics.json` kèm `git_sha` · `seed` · `config_sha256` · **`n_molecules` (= 932, KHÔNG phải số
ảnh)**. Trường cuối là để không ai lỡ báo CI trên số ảnh — khoảng tin cậy sẽ hẹp sai √n_view lần
(cảnh báo ở M7).

Dùng `metrics.json` chứ không chỉ markdown, để `04_ablate_views.py` và `05_ablate_resolution.py`
gom kết quả mà không phải parse văn bản.

---

## 12.7 ⚠ Đính chính §5 — ba mốc so sánh không cùng split

Bảng cuối §5 ghi: *"dự báo hằng số 1,091 · ECFP4+descriptor 0,751 · Chemprop D-MPNN 0,674"*.
**Chỉ con số đầu đo trên `split_all.csv`.** Hai số còn lại:

| Mốc | Vấn đề |
|---|---|
| **1,091** | ✅ Hợp lệ — đo trên `split_all.csv` (§3) |
| ~~0,751~~ | Đo trên 3 split `scaffold_balanced`, và **chính §2.8 đã tự đính chính thành 0,780 ± 0,056** (5 split). 0,751 là số đã bị bác bỏ trong cùng tài liệu |
| ~~0,674~~ | Chemprop, split riêng, test N=935 ≠ 932 chất của `split_all.csv` |

Với nhiễu giữa các split = **0,056** (§11.7), đặt kết quả model ảnh cạnh 0,674 là phép so hỏng —
đúng kiểu hỏng mà §2.9 và K.2.1 đã cảnh báo. §11.7 cũng đã ghi: *"RMSE tuyệt đối không so được với
bất kỳ số nào tính trên split khác."*

**Mốc hợp lệ trên `split_all.csv`:**

| Baseline | Nguồn | Trạng thái |
|---|---|---|
| Dự báo hằng số | **1,091** | ✅ đã có (§3) |
| ECFP4 + descriptor | chạy lại `baselines/ecfp_gbm.py` trên `split_all.csv`, 5 seed khởi tạo | cần chạy |
| Chemprop D-MPNN | — | **bỏ khỏi đường chạy** (quyết định 2026-10-27) |

`ecfp_gbm.py` **không phải tùy chọn**: M8 có bootstrap ghép cặp *"so với baseline tô-pô trên cùng
932 chất"*, và phép đó cần `y_pred` mức phân tử của một baseline tô-pô trên đúng tập test này.
Chạy một lần, vài phút, xong.

**Schema chung cho mọi baseline** (để M8 không phải viết parser riêng):
`runs/baseline_<tên>/predictions_test.csv` với cột `smiles, y_true, y_pred`, mức phân tử, 932 dòng.

---

## 12.8 Ba quyết định phải có tên trường trong `configs/base.yaml`

Không phải thiếu file — là thiếu quyết định mà chữ ký hàm của M4–M7 phụ thuộc vào.

| # | Quyết định | Chốt | Lý do |
|---|---|---|---|
| 1 | **V1 "chính diện" là mặt phẳng nào** | **PC1–PC2** | Hai trục trải rộng nhất; §8.1 ghi V3 = PC1-PC2, PC1-PC3, PC2-PC3 → V1 trùng view đầu của V3, nên `04_ablate_views.py` so được trực tiếp |
| 2 | **Median qua ảnh khi `n_view=1`** | Viết tổng quát, **không** làm trường hợp đặc biệt | V27 dùng lại đúng hàm đó. Median của một phần tử là chính nó — đúng, chỉ cần code không rẽ nhánh |
| 3 | **TTA áp ở đâu** | **Bật cả val lẫn test** | Nếu TTA chỉ bật lúc test thì early-stop đang chọn checkpoint theo một đại lượng **khác** với đại lượng báo cáo. Giữ hai đại lượng trùng nhau |

---

## 12.9 `.gitignore` — chỗ dễ sai vì có file phải lọt qua

```gitignore
data/raw/*.xlsx
data/interim/*
data/renders/*
runs/*

!data/raw/split_all.csv
!data/interim/geometry.parquet
!data/renders/*/
!data/renders/*/MANIFEST.json
!data/renders/*/index.parquet
!data/renders/*/qa/
!runs/*/
!runs/*/config.yaml
!runs/*/metrics.json
!runs/*/*/metrics.json
!runs/*/*/train_log.csv
!runs/baseline_*/predictions_test.csv

*.ckpt
*.npy
__pycache__/
.ipynb_checkpoints/
```

⚠ **Git không đi vào thư mục đã bị ignore**, nên hai dòng `!data/renders/*/` và `!runs/*/` là
**bắt buộc** — thiếu chúng thì mọi negation bên dưới đều vô hiệu và `MANIFEST.json` sẽ lặng lẽ
không được commit.

Xác nhận trước khi tin:

```bash
git check-ignore -v data/renders/v1/MANIFEST.json   # phải KHÔNG khớp luật nào
git check-ignore -v data/interim/geometry.parquet   # phải KHÔNG khớp luật nào
```

---

## 12.10 Thứ tự viết code

| # | Việc | Vì sao thứ tự này |
|---|---|---|
| 0 | **Kiểm `pymol2`** (§12.0) | Có thể vứt bỏ `m3_render.py` — kiểm trước khi viết thêm |
| 1 | `configs/base.yaml` · `render_v1.yaml` · `model_r18.yaml` | Ba quyết định ở §12.8 hóa thành tên trường; M4–M7 đọc thẳng từ đây |
| 2 | `scripts/01_build_conformers.py` | Dùng M1+M2 đã có → cả 3 file `data/interim/`. **Nhớ `flat_ratio` + `has_undefined_stereo`** (§12.4) |
| 3 | `scripts/00_audit_split.py` + `baselines/constant.py` | Rẻ, và chặn đường phép so sánh cuối cùng |
| 4 | `src/m4_dataset.py` → `m5_model.py` → `m6_train.py` → `m7_eval.py` → `scripts/03_train.py` | Cụm chính, viết một lượt |
| 5 | `baselines/ecfp_gbm.py` trên `split_all.csv` | Mốc tô-pô cho bootstrap ở M8 |
| 6 | `tests/test_align_deterministic.py` · `test_no_mirror.py` · `test_split_integrity.py` | Trước khi chạy 3 seed |
| 7 | **Bước 1 của §8.5**: V1 + ResNet-18 + ImageNet, 3 seed | Con số đầu tiên |

Chưa cần ở giai đoạn này: `m8_report.py` · `04`–`07` · `configs/model_effb0` · `configs/ablation/*` ·
toàn bộ `reports/` ngoài `split_audit.md`. §9.1 nói bắt đầu từ ResNet-18 và chỉ tăng dung lượng
nếu **train RMSE** cao — nên viết sẵn ba config backbone là viết thừa hai.
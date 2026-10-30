# Thu tu chay: env -> check -> conformers -> render -> baselines -> train -> (chot cau hinh) -> final-test
PY ?= python
RUN ?= runs/v1_r18

.PHONY: env check conformers render sheet baselines train final-test test audit
env:
	conda env create -f environment.yml

check:
	bash scripts/00_check_pipeline.sh

audit:
	$(PY) scripts/00_audit_split.py

conformers:
	$(PY) scripts/01_build_conformers.py

render:
	$(PY) scripts/02_render.py --config configs/base.yaml configs/render_v1.yaml

sheet:
	$(PY) scripts/09_contact_sheet.py --config configs/base.yaml configs/render_v1.yaml

baselines:
	$(PY) baselines/constant.py
	$(PY) baselines/ecfp_gbm.py

train:
	$(PY) scripts/03_train.py

test:
	$(PY) -m pytest tests -q

# Chi chay MOT lan cho cau hinh da chot: make final-test REASON="..."
final-test:
	$(PY) scripts/08_final_test.py $(RUN) --reason "$(REASON)"

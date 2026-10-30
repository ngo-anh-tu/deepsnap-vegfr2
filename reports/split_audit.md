# Audit split_all.csv

Sinh boi `scripts/00_audit_split.py` -- khong chep tay. "LECH" = khac so ghi o §3.

| Muc | Do duoc | Ky vong (§3) | |
|---|---|---|---|
| scaffold xuyen split | 0 | 0 | OK |
| so scaffold | 3521 | 3521 | OK |
| lech mean test-train | -0.063 | -0.063 | OK |
| RMSE hang so (mean train) | 1.091 | 1.091 | OK |
| scaffold khop quy uoc includeChirality=True (%) | 100.0 | 100.0 | OK |
| so chat test | 932 | 932 | OK |
| chat bi loai so voi 9.341 | n/a (can --clean) | 35 | chua kiem |
| cap enantiomer | 75 | 75 | OK |
|   - bi tach qua split | 10 | 10 | OK |
|   - trong do train<->test | 3 | 3 | OK |

Phan bo split: train=7444, test=932, val=930

# 터보팬 엔진 잔여 수명(RUL) 예측 — NASA C-MAPSS FD001


항공기 터보팬 엔진의 센서 시계열로 **잔여 수명(RUL, Remaining Useful Life)** 을 예측하는 예지보전(Predictive Maintenance) 프로젝트입니다. NASA C-MAPSS FD001(엔진 100대, 21개 센서, run-to-failure)을 사용해 고장 시점을 사전 예측하고 정비 의사결정을 지원합니다.

> **진행 원칙:** 바로 모델링하지 않습니다. `문제정의 → 데이터이해(EDA) → 전처리 → 모델링 → 평가 → 개선` 순서로 진행하며, 각 단계마다 결정·근거를 문서로 남기고, AI 리뷰를 활용해 문제 정의와 결과 해석을 점검했습니다. 단계별 인덱스는 [STEP_INDEX.md](STEP_INDEX.md) 참고.

## 현재 프로젝트의 성격

본 프로젝트는 NASA C-MAPSS FD001 데이터를 활용해 RUL 예측 문제를 정의하고,
데이터 로딩 → RUL 생성 → EDA → 전처리 → 모델링 → 평가 → 시계열 피처 개선 → threshold 분석까지
하나의 예지보전 분석 흐름을 완주한 파이프라인 레퍼런스입니다.

다만 본 결과는 FD001에서의 최적 모델 탐색을 완료한 결과라기보다는,
향후 XGBoost, LightGBM, validation 기반 threshold optimization, GroupKFold 검증을 추가하기 위한
baseline reference로 보는 것이 적절합니다.

## 1. 문제 정의

- **비즈니스 목표**: 항공사 정비 계획 담당자가 엔진 고장 시점을 사전에 알지 못해 발생하는 과잉 예방 정비 비용과 예상치 못한 고장(결항·안전사고)을 줄인다.
- **ML 목표**: 센서·운전조건 시계열로 각 시점의 RUL(사이클 수)을 **회귀** 예측. RUL ≤ 30 구간은 **조기 경보(분류)** 로 병행.
- **성공 기준**: RMSE ≤ 30, 조기경보 Recall ≥ 85%, FNR ≤ 5%.
- **오판정 비용**: False Negative(고장 임박 미탐지) ≫ False Positive(과잉 정비). FN은 결항·안전사고로 직결.

자세히: [docs/01_problem_definition.md](docs/01_problem_definition.md)

## 2. 데이터

- 출처: NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)
- 규모: Train 100엔진 / 20,631행, Test 100엔진 / 13,096행, 26컬럼(엔진ID, cycle, 운전조건 3, 센서 21)
- 타깃: RUL = `engine별 max_cycle − 현재 cycle` (train에서 생성, engine 1 첫 행 = 191 검증)
- 조건: 단일 운전조건(Sea Level), 단일 고장모드(HPC Degradation)
- 로딩 함정: 헤더 없음 + 행 끝 공백 2개 → `sep=r'\s+'` 필수(26컬럼)

## 3. 전체 진행 흐름

| Step | 단계 | 핵심 결과 |
|---|---|---|
| 01 | 세팅 + RUL 생성 | 구조·문제정의서·로딩·RUL 생성 (engine1 첫행 191) |
| 02 | EDA | 무변화 센서 식별, RUL 상관 센서 파악, 수명 128~362 |
| 03 | 전처리 | 무변화 센서 7개 제거, MinMax(train fit), RUL_clipped(cap=125) |
| 04 | 모델링 | baseline→회귀 비교. RF/GB RMSE~18, 단 FNR 0.28 |
| 05 | 평가 & 개선 | 위험구간 과대예측(+9.42)·short-life FN 진단 |
| 06 | 개선 실험 | 시계열 피처로 위험구간 과대예측 완화, 성공기준 충족 |
| 07 | 최종 문서화 | FD001 레퍼런스 정리 + FD002~004 확장 계획 |

## 4. 주요 EDA 결과

- 엔진 수명 128~362 사이클(평균 약 206, 중앙값 199) — 편차 큼.
- 무변화 센서: 완전상수 sensor_1·10·18·19 / 준상수 sensor_5·6·16 → 제거 후보 7개.
- RUL 상관 상위: sensor_11(-0.70), 4(-0.68), 12(+0.67), 7(+0.66) 등.
- 상관계수는 참고 지표일 뿐, 시계열 추세·모델 기반 중요도와 함께 해석.

자세히: [docs/02_eda_summary.md](docs/02_eda_summary.md)

## 5. 전처리 기준

- 무변화 센서 7개 제거(완전상수 4 + 준상수 3) → 유효 센서 14개.
- op_setting 제외(FD001 단일조건 전용 결정 — FD002~004에선 재검토).
- MinMax 정규화: **train으로만 fit**, test는 transform만(누수 방지).
- Piecewise Linear RUL: cap=125, 원본 RUL과 RUL_clipped 둘 다 보존.

자세히: [docs/03_preprocessing_summary.md](docs/03_preprocessing_summary.md)

## 6. 모델 성능 (test: 엔진별 마지막 cycle, true RUL 기준)

### 기본 모델 (현재 센서값, Step 04)

| 모델 | RMSE | Alert Recall(t=30) | FNR(t=30) |
|---|---|---|---|
| Baseline(평균) | 43.07 | 0.00 | 1.00 |
| Linear/Ridge | 21.69 | 0.36 | 0.64 |
| RandomForest | 18.16 | 0.72 | 0.28 |
| GradientBoosting | 18.39 | 0.72 | 0.28 |

→ RMSE 기준은 충족했으나 조기경보 FNR(0.28)이 목표(≤5%)에 크게 미달.

### 개선 모델 (시계열 피처, Step 06)

| 지표 (RandomForest) | 기본(S04) | 시계열(S06) |
|---|---|---|
| RMSE | 18.16 | **14.67** |
| NASA Score | 951 | **356** |
| 위험구간(RUL≤30) 평균 잔차 | +9.42 | **+3.73** |

threshold별(RandomForest_ts): t=30 FNR 0.28 → **t=40 FNR 0.00, Recall 1.00 (FP 3)**.

자세히: [docs/04_modeling_summary.md](docs/04_modeling_summary.md), [docs/05_evaluation_summary.md](docs/05_evaluation_summary.md), [docs/06_improvement_summary.md](docs/06_improvement_summary.md)

## 7. 현재 기준 모델 및 운영 threshold 후보

> **RandomForest + 시계열 피처, 조기경보 threshold = 40 (FD001 기준 후보 운영값)**

| 지표 | 값 |
|---|---|
| RMSE | 14.67 |
| NASA Score | 356 |
| Precision | 0.893 |
| Recall | 1.00 |
| FNR | 0.00 |
| FP / FN | 3 / 0 |

FD001 test 기준에서 RandomForest + 시계열 피처 모델은 RMSE 기준을 만족하였다.
조기경보 성능은 기본 threshold=30에서는 FNR 0.28로 부족했으나,
운영 threshold를 40으로 조정하면 Recall 1.00, FNR 0.00을 달성하였다.
따라서 threshold=40은 FD001 기준 후보 운영값으로 볼 수 있다.

단, threshold=40은 별도 validation set에서 선택한 값이 아니므로
최종 일반화된 최적 threshold로 단정하지 않는다.

## 8. 비즈니스 해석

기본 모델은 RMSE는 좋았지만 고장 임박 엔진을 25대 중 7대 놓쳐(FNR 0.28) 그대로 쓰기 어려웠습니다. 시계열 피처(열화 속도)를 추가하자 위험 구간 과대 예측이 절반 이하로 줄었고, 운영 임계값을 40으로 두면 고장 임박 엔진을 전부 잡으면서(Recall 1.00) 오경보는 3건에 그칩니다. 예지보전에서 FN 비용이 FP보다 크므로, 약간의 과잉 정비를 감수하고 미탐지를 0으로 만드는 이 균형은 합리적입니다.

## 9. 한계

- **FD001 test 100대 기준** 결과로, 일반화를 보장하지 않습니다.
- 기본 threshold=30에서는 RandomForest FNR이 여전히 0.28 — 개선은 "threshold=40 운영값과 결합했을 때" 성립.
- 시계열 피처로 feature가 98개로 늘어 **과적합 가능성**이 있습니다. 독립 단위는 엔진 100대.
- 엔진 단위 교차검증(GroupKFold)·validation split 등 **안정성 추가 검증은 미수행**.
- threshold=40은 **FD001 기준 후보 운영값**이며 확정값이 아닙니다.

## 10. FD002~FD004 확장 계획

FD001은 단일 운전조건·단일 고장모드입니다. FD002~004는 운전조건·고장모드가 달라 **이 프로젝트의 결정을 그대로 적용하면 안 됩니다.**

- op_setting을 제거하지 않고 모델 입력/조건별 정규화로 활용
- 무변화 센서 목록을 그대로 쓰지 말고 FD별 EDA로 재결정
- threshold=40을 그대로 적용하지 말 것
- 각 FD별로 Step 01~06 흐름을 반복

자세히: [docs/FD002_003_004_extension_plan.md](docs/FD002_003_004_extension_plan.md), 표준 양식: [docs/FD001_reference_workflow.md](docs/FD001_reference_workflow.md)

## 데이터 준비

본 레포에는 원본 데이터와 전처리 CSV를 포함하지 않습니다.  
NASA C-MAPSS FD001 데이터를 다운로드한 뒤 아래 위치에 배치합니다.

```text
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

이후 아래 재현 방법의 명령어를 순서대로 실행하면 `data/processed/` 산출물이 생성됩니다.

## 11. 재현 방법

```bash
pip install -r requirements.txt

# src 모듈을 순서대로 실행 (각 단계 산출물 생성)
python3 src/make_rul_dataset.py        # data/processed/train_with_rul.csv
python3 src/preprocess.py              # train/test_preprocessed.csv
python3 src/modeling.py                # reports/model_results.csv
python3 src/evaluation_analysis.py     # reports/threshold_analysis.csv, fn_engine_analysis.csv
python3 src/feature_engineering.py     # train/test_features.csv (누수 점검 포함)
python3 src/improvement_modeling.py    # reports/improvement_results.csv

# 노트북: notebooks/ 안에서 01 → 02 → 03 → 04 순서로 실행
```

## 레포 구조

```text
cmapss-rul-project/
├── README.md
├── STEP_INDEX.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # 원본 데이터 배치 위치 (.gitkeep만 업로드)
│   └── processed/      # 전처리 산출물 저장 위치 (.gitkeep만 업로드)
├── notebooks/          # 01_eda → 02_modeling → 03_evaluation → 04_improvement
├── src/                # 데이터 로딩, RUL 생성, 전처리, 모델링, 평가 코드
├── models/             # 모델/스케일러 저장 위치 (.pkl은 gitignore)
├── reports/            # 성능·threshold·FN 분석 결과 CSV
├── docs/               # 문제정의, EDA, 전처리, 모델링, 평가, 개선, 확장 계획 문서
├── ppt/                # 발표자료 위치
└── ref/                # 참고자료
```

> `reports/final_model_summary.csv`는 Step 07 기준 현재 레퍼런스 결과 요약이며, FD001 최적 모델 확정 결과를 의미하지 않습니다.

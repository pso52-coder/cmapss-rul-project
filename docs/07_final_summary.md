# 최종 요약 — NASA C-MAPSS FD001 RUL 예측 (Step 07)

> FD001 레퍼런스 프로젝트의 전체 요약. 추가 튜닝이 아니라 결과 정리와 확장 기반 마련이 목적.

## 1. 프로젝트 개요

항공기 터보팬 엔진의 센서 시계열로 잔여 수명(RUL)을 예측하는 예지보전 프로젝트. 고장 시점을 사전에 알려 정비 의사결정을 지원하고, 과잉 예방 정비와 예상치 못한 고장(결항·안전사고)을 줄이는 것이 목표.

- 성공 기준: RMSE ≤ 30, 조기경보 Recall ≥ 85%, FNR ≤ 5%
- 핵심 원칙: 단계를 건너뛰지 않고 각 단계 검증 + GPT 교차 검토. FN(고장 임박 미탐지)이 가장 위험.

## 2. Step 01~06 핵심 결과

| Step | 단계 | 핵심 결과 |
|---|---|---|
| 01 | 세팅 + RUL 생성 | 구조·문제정의서 작성. RUL = max_cycle−cycle, engine1 첫행 191 검증. sep=r'\s+'로 26컬럼 로딩. |
| 02 | EDA | 엔진 수명 128~362. 무변화 센서 7개 식별. RUL 상관 상위 sensor_11/4/12/7. |
| 03 | 전처리 | 무변화 센서 7개 제거(유효 14개), op_setting 제외, MinMax(train fit), RUL_clipped(cap=125). |
| 04 | 모델링 | baseline→회귀. RF RMSE 18.16, 단 FNR 0.28(목표 미달). |
| 05 | 평가 & 개선 | 위험구간 과대예측(평균잔차 +9.42)·short-life FN 진단. threshold만으론 한계. |
| 06 | 개선 실험 | 시계열 피처로 위험구간 잔차 +3.73. RF_ts+threshold=40에서 FNR 0.00·Recall 1.00. |

## 3. 최종 성능 (FD001 test 100대 기준)

> **최종 선택: RandomForest + 시계열 피처, 조기경보 threshold = 40 (FD001 기준 후보 운영값)**

| 지표 | 값 | 성공 기준 | 달성 |
|---|---|---|---|
| RMSE | 14.67 | ≤ 30 | O |
| NASA Score | 356 | 최소화 | (baseline 33,629 대비 대폭↓) |
| Recall (t=40) | 1.00 | ≥ 0.85 | O |
| FNR (t=40) | 0.00 | ≤ 0.05 | O |
| Precision (t=40) | 0.893 | — | 참고 |
| FP / FN | 3 / 0 | — | 참고 |

## 4. 성공 기준 달성 여부

FD001 test 기준에서 RMSE는 목표 기준을 충족하였다.
조기경보 지표는 기본 threshold=30에서는 Recall 0.72, FNR 0.28로 목표에 미달하였다.

다만 예지보전에서는 False Negative 비용이 False Positive보다 크기 때문에,
운영 threshold를 40으로 보수적으로 조정했을 때 Recall 1.00, FNR 0.00을 달성하였다.

따라서 본 결과는 “RandomForest_ts 모델 자체가 threshold=30에서 모든 기준을 만족했다”기보다는,
“RandomForest_ts + FD001 기준 후보 운영 threshold=40 조합이 조기경보 목표를 만족했다”로 해석하는 것이 적절하다. 단, 다음을 전제로 한다.
- 기본 threshold=30이 아니라 **운영 threshold=40**과 결합했을 때 성립.
- 시계열 피처가 위험 구간 과대 예측을 완화한 것이 결정적이었음.

## 5. 최종 결론

본 프로젝트의 최종 산출물은 FD001 최적 모델이라기보다,
FD001 RUL 예측 파이프라인을 완주한 baseline reference이다.
후속 단계에서는 XGBoost, LightGBM, sample weighting, validation 기반 threshold optimization을 통해
FD001 내부에서 모델링 레벨을 추가로 고도화한 뒤 FD002~FD004로 확장하는 것이 적절하다.

## 6. 한계와 다음 과제

**한계**
- FD001 test 100대 기준 결과 — 일반화 보장 아님.
- 기본 threshold=30에서는 RF FNR 여전히 0.28. 개선은 threshold=40 결합 시 성립.
- feature 98개로 증가 → 과적합 가능성. 독립 단위는 엔진 100대.
- 엔진 단위 교차검증(GroupKFold)·validation split 미수행.
- threshold=40은 FD001 기준 후보값(확정 아님).

**다음 과제 (선택)**
- 엔진 단위 GroupKFold로 threshold·성능 안정성 검증.
- 하이퍼파라미터 튜닝(목표=위험구간 Recall).
- FD002~FD004 확장 ([docs/FD002_003_004_extension_plan.md](FD002_003_004_extension_plan.md)).

## 7. 산출물 인덱스

- 문서: docs/01~07, FD001_reference_workflow.md, FD002_003_004_extension_plan.md
- 코드: src/ (data_loader, make_rul_dataset, preprocess, modeling, evaluation_analysis, feature_engineering, improvement_modeling)
- 노트북: notebooks/01_eda ~ 04_improvement_experiment
- 리포트: reports/ (model_results, threshold_analysis, fn_engine_analysis, improvement_results, threshold_analysis_improved, final_model_summary)
- 단계 기록: STEP_INDEX.md 및 docs/01~08 문서에 단계별 판단 근거 요약

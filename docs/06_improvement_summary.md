# 개선 실험 요약 — 시계열 피처 (Step 06)

> 코드: `src/feature_engineering.py`, `src/improvement_modeling.py`
> 노트북: `notebooks/04_improvement_experiment.ipynb`
> 결과: `reports/improvement_results.csv`, `reports/threshold_analysis_improved.csv`
> **목적:** Step 05 약점(위험 구간 과대 예측, short-life FN)을 시계열 피처로 개선. 새 모델 추가가 아니라 같은 RF/GB에 "열화 속도" 정보 추가.

## 1. 시계열 피처

유효 센서 14개 각각에 대해 생성 (총 98개 피처):
- `_ma5`, `_ma10`: 최근 5/10 cycle 이동평균
- `_slope5`, `_slope10`: 최근 5/10 cycle 선형 기울기 (열화 속도)
- `_delta_init`: 초기값 대비 변화량
- `_dev_ma5`: 현재값 − 최근 이동평균

**데이터 누수 방지:** 각 row 피처는 그 cycle의 과거+현재 값만 사용(미래 미사용). engine_id별 시간순 계산. 누수 점검 통과(엔진 1에서 마지막 cycle 제거 후 재계산해도 피처 일치, max_diff=0.0). test도 동일 방식.

**수명 구간 정의:** train max_cycle 분포의 하위 33% = short / 중간 33% = mid / 상위 33% = long.

## 2. 성능 비교 (Step 04 현재값 vs Step 06 시계열)

| 지표 | RandomForest S04 | RandomForest S06 | GradientBoosting S04 | GradientBoosting S06 |
|---|---|---|---|---|
| RMSE | 18.16 | **14.67** | 18.39 | **14.99** |
| MAE | 13.24 | 10.76 | 13.09 | 11.00 |
| NASA Score | 951 | **356** | 1144 | 395 |
| Alert Recall (t=30) | 0.72 | 0.72 | 0.72 | **0.80** |
| FNR (t=30) | 0.28 | 0.28 | 0.28 | **0.20** |

→ RMSE·MAE·NASA Score 모두 개선. GradientBoosting은 기본 threshold(30)에서도 Recall/FNR 개선.

## 3. 위험 구간 과대 예측 완화 (핵심 성과)

| 지표 (RandomForest) | Step 05 | Step 06 |
|---|---|---|
| 고장임박 구간(RUL≤30) 평균 잔차 | +9.42 | **+3.73** |
| 위험구간 과대 예측 수 | 20/25 | 18/25 |

→ 시계열 피처가 겨냥한 약점이 직접 완화됐다. 위험 구간 과대 예측 잔차가 절반 이하로 줄었다.

## 4. threshold 확장 분석 (RandomForest_ts, 30~70)

| threshold | Precision | Recall | F1 | FNR | FP | FN |
|---|---|---|---|---|---|---|
| 30 | 0.947 | 0.72 | 0.818 | 0.28 | 1 | 7 |
| **40** | 0.893 | **1.00** | 0.943 | **0.00** | 3 | 0 |
| 50 | 0.833 | 1.00 | 0.909 | 0.00 | 5 | 0 |
| 60 | 0.676 | 1.00 | 0.806 | 0.00 | 12 | 0 |
| 65 | 0.658 | 1.00 | 0.794 | 0.00 | 13 | 0 |
| 70 | 0.641 | 1.00 | 0.781 | 0.00 | 14 | 0 |

**핵심:** Step 05에서는 threshold=60까지 올려도 FNR=0.08(FP 11)이었다. 시계열 피처 후에는 **threshold=40만으로 FNR=0.00, Recall=1.00을 FP 3건만으로** 달성한다. 즉 모델이 위험 구간을 훨씬 정확히 예측하게 되어, 작은 운영 완충(40)만으로 목표를 충족한다.

## 5. FN 엔진 변화

threshold=30 기준 FN 엔진 수는 RF에서 7대로 동일하지만, **오차 크기가 급감**했다.

| | Step 05 FN 오차 범위 | Step 06 FN 오차 범위 |
|---|---|---|
| RandomForest | +7 ~ +49 | +3 ~ +17 |

→ 놓친 엔진들도 "겨우 경계를 넘긴" 수준으로 바뀌어, threshold=40에서 모두 잡힌다. FN vs TP 엔진 비교에서도 열화 속도 피처(기울기·초기 대비 변화량)에 차이가 나타나, 시계열 정보가 빠른 열화 신호를 일부 포착했음을 확인했다.

## 6. 결론 (과장 없음)

- 시계열 피처는 RMSE·NASA Score를 개선하고, 목표했던 **위험 구간 과대 예측을 절반 이하로 완화**했다.
- **시계열 피처(RandomForest) + threshold=40** 조합은 이 test set에서 RMSE 14.67, FNR 0.00, Recall 1.00(FP 3)으로 **성공 기준(RMSE≤30, Recall≥85%, FNR≤5%)을 모두 충족**한다.
- 단, 이는 **FD001 test 100대 기준 결과**이며 일반화를 보장하지 않는다. 운영값(threshold)은 FP 비용을 감안한 정책 결정으로 남긴다.
- 다음 가능 단계(선택): 모델 튜닝, 교차검증으로 안정성 확인, README 결과 정리, FD002~004 확장.

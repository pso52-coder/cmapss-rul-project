# 모델링 요약 — NASA C-MAPSS FD001 (Step 04)

> 입력: `data/processed/train_preprocessed.csv`, `data/processed/test_preprocessed.csv`, `data/raw/RUL_FD001.txt`
> 코드: `src/modeling.py` / 노트북: `notebooks/02_modeling.ipynb` / 결과: `reports/model_results.csv`
> **범위:** baseline + 기본 회귀 모델 비교까지. 하이퍼파라미터 튜닝·딥러닝·시계열 모델은 이후 단계.

## 1. 평가 설계 (중요)

- **학습 타깃:** `RUL_clipped` (cap=125)
- **test 평가 방식:** test 전체 행이 아니라 **engine_id별 마지막 cycle 행**(엔진 100대 × 1행)만 추출 → `RUL_FD001.txt`의 true RUL과 결합해 비교.
- **clipped vs 원본 기준 구분:** 학습 타깃이 clipped(최대 125)이므로 예측값도 [0, 125]로 클리핑해 평가했다. true RUL이 125를 넘는 일부 엔진은 구조적으로 오차가 생기지만, 이는 "고장 임박 구간을 잘 잡는다"는 목표상 감수하는 트레이드오프다. (원본 RUL 정밀 예측이 목적이 아님)
- **지표:** RMSE, MAE, NASA Score(늦은 예측에 큰 패널티), 그리고 **RUL≤30 조기경보 Recall / FNR**.

## 2. 모델별 성능 (test, true RUL 기준)

| 모델 | RMSE | MAE | NASA Score | Alert Recall | FNR | FN 개수 |
|---|---|---|---|---|---|---|
| Baseline (평균) | 43.07 | 35.90 | 33629 | 0.00 | 1.00 | 25 |
| Baseline (cycle LR) | 33.34 | 27.59 | 5319 | 0.04 | 0.96 | 24 |
| LinearRegression | 21.69 | 17.44 | 1288 | 0.36 | 0.64 | 16 |
| Ridge | 21.69 | 17.44 | 1285 | 0.36 | 0.64 | 16 |
| **RandomForest** | **18.16** | 13.24 | 951 | **0.72** | **0.28** | 7 |
| GradientBoosting | 18.39 | 13.09 | 1144 | 0.72 | 0.28 | 7 |

(조기경보 분모: 실제 RUL≤30인 엔진 25대)

## 3. 해석

- **Baseline의 역할:** 평균 예측은 조기경보를 하나도 못 잡고(Recall 0), cycle 선형회귀도 거의 못 잡는다. "ML 모델이 최소한 이걸 넘어야 한다"는 기준선으로서, 센서 기반 모델이 이를 크게 능가함을 확인했다.
- **선형 → 트리:** Linear/Ridge는 RMSE 21.7로 baseline을 넘지만, 비선형 열화 패턴을 충분히 못 잡는다. RandomForest/GradientBoosting이 RMSE 약 18로 가장 우수하고, 조기경보 Recall도 0.72로 가장 높다.
- **RMSE 성공 기준:** 문제 정의의 RMSE ≤ 30 기준을 트리 모델(18대)과 선형 모델(21대) 모두 충족한다.
- **가장 중요한 한계 — FNR:** 최고 모델조차 FNR이 0.28이다. 즉 실제 고장 임박 엔진 25대 중 7대를 "정상"으로 놓쳤다. 이 프로젝트의 성공 기준(FNR ≤ 5%)에는 **아직 도달하지 못했다.** 예지보전에서 FN은 결항·안전사고로 직결되므로, 현재 모델을 그대로 현장에 쓰기에는 부족하다.

> 결론을 과장하지 않는다: RMSE 기준으로는 성공 범위지만, **비즈니스상 핵심인 FN 관점에서는 미완성**이다.

## 4. 모델별 장단점

| 모델 | 장점 | 단점 |
|---|---|---|
| Baseline(평균/cycle) | 기준선 제공 | 조기경보 거의 못 잡음 |
| Linear / Ridge | 빠름·해석 쉬움 | 비선형 패턴 한계 |
| RandomForest | RMSE·Recall 최고, 비선형 포착 | 외삽 약함 |
| GradientBoosting | MAE 최저, RF와 유사 | 학습 느림, 튜닝 민감 |

## 5. 다음 단계 제안 (Step 05 평가/개선, 이 단계 미수행)

1. **FN 줄이기:** 조기경보 임계값을 RUL=30보다 보수적으로(예: 예측 RUL≤40이면 경보) 조정해 Recall을 끌어올리고 FNR을 목표에 맞춘다. (Recall↔과검사 트레이드오프 분석)
2. **시계열 피처:** 이동평균·기울기 등 열화 추세 피처를 추가해 마지막 시점 예측 정확도 향상.
3. **준상수 센서 버전 비교:** 완전상수만 제거한 버전(센서 17개)과 성능 비교.
4. **하이퍼파라미터 튜닝** 및 모델 해석(피처 중요도).
5. 평가 결과를 README의 "결과" 섹션과 비즈니스 번역 한 문단으로 정리.

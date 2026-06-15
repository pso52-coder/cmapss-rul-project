# 전처리 요약 — NASA C-MAPSS FD001 (Step 03)

> 입력: `data/processed/train_with_rul.csv`, 원시 `data/raw/test_FD001.txt`
> 출력: `data/processed/train_preprocessed.csv`, `data/processed/test_preprocessed.csv`
> 코드: `src/preprocess.py`
> **범위:** 모델링에 사용할 학습용 데이터셋 생성까지. 모델 학습·평가는 다음 단계(Step 04)로 분리.

## 1. 무변화 센서 제거

EDA에서 도출한 무변화 센서를 제거했다. **완전 상수와 준상수를 구분해서 기록**한다.

| 분류 | 센서 | std 기준 | 처리 |
|---|---|---|---|
| 완전 상수 | sensor_1, sensor_10, sensor_18, sensor_19 | std = 0 | **제거 확정** (정보 없음) |
| 준상수 | sensor_5, sensor_6, sensor_16 | 0 < std < 0.01 | **제거 후보** (기본 전처리에서 함께 제거) |

- 준상수 센서는 임계값(0.01)이 센서 단위·스케일의 영향을 받을 수 있어, 원칙적으로는 "제거 후보"다. 다만 본 학습 흐름에서는 기본 전처리 데이터에서 7개를 모두 제거했다.
- 추후 성능 비교가 필요하면 두 버전을 만들 수 있다: **버전 A**(완전 상수 4개만 제거) vs **버전 B**(완전 상수+준상수 7개 제거, = 현재 기본).
- **제거: 7개 / 남은 유효 센서: 14개**
  - 남은 센서: sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21

## 2. 운전조건(op_setting) 제외

FD001은 단일 운전 조건이라 op_setting_1~3이 거의 상수다(std: 0.0022 / 0.0003 / 0.0). 변별 정보가 없어 피처에서 제외했다.

> **확장성 주의 (FD001 전용 결정):** op_setting 제외는 FD001(단일 운전조건)에만 적용되는 결정이다. FD002~FD004는 운전 조건이 다양해지므로, 그때는 op_setting을 제거하지 말고 모델 입력으로 쓰거나 조건별 정규화 기준으로 활용해야 한다.

## 3. 센서 정규화 (Min-Max) — 데이터 누수 방지

- 방식: `MinMaxScaler` (각 센서를 [0, 1]로). 센서별 스케일·범위가 크게 달라 정규화 없이는 일부 센서가 과도한 영향을 준다.
- **누수 방지 원칙:** scaler는 **train으로만 fit**하고, test에는 **transform만 적용**했다. test 기준으로 fit하지 않는다.
- 검증: train 피처 범위는 정확히 [0.000, 1.000]. test 피처 범위는 [-0.044, 1.081]로 0~1을 약간 벗어나는데, 이는 train 기준으로 변환했다는 증거이며 정상이다(test에 train에 없던 값이 있을 수 있음).

## 4. Piecewise Linear RUL 클리핑

- 상한값: **cap = 125**
- 근거:
  - C-MAPSS 문헌에서 가장 널리 쓰이는 표준값(125~130).
  - 엔진 최소 수명이 128이므로 cap=125는 **어떤 엔진도 전 구간이 상한에 묶이지 않는다**(전 구간 평탄화 방지). cap=130은 최소 수명 엔진 1대가 전 구간 클리핑되는 문제 발생.
  - 선형 RUL은 초기 구간(열화 거의 없음)을 비현실적으로 크게 라벨링하므로, 일정 값 이상을 상한으로 고정해 "열화가 시작된 이후" 신호에 집중하게 한다.
- **원본 RUL과 clipped RUL을 둘 다 저장:**
  - `RUL` — 원본 (max_cycle − cycle)
  - `RUL_clipped` — min(RUL, 125)
- 검증: engine 1 첫 행 `RUL=191`, `RUL_clipped=125` (상한 적용 확인). 클리핑 후 최대값 = 125.

## 5. 출력 파일 구조

| 파일 | shape | 컬럼 |
|---|---|---|
| train_preprocessed.csv | (20631, 18) | engine_id, cycle, 정규화 센서 14개, RUL, RUL_clipped |
| test_preprocessed.csv | (13096, 16) | engine_id, cycle, 정규화 센서 14개 |

- test에는 행별 RUL을 만들지 않는다. test의 정답 RUL은 각 엔진의 **마지막 cycle 시점** 값으로 `data/raw/RUL_FD001.txt`에 별도 제공되며, 이는 Step 04 평가에서 사용한다. (행별 RUL을 임의로 만들면 평가 기준이 왜곡됨)

## 6. 검증 요약

| 항목 | 결과 | 판단 |
|---|---|---|
| 유효 센서 수 | 14 | PASS |
| 제거 센서 수 | 7 (완전4 + 준상수3) | PASS |
| train 피처 범위 | [0.0, 1.0] | PASS |
| test 피처 범위 | [-0.044, 1.081] | 정상(누수 방지 증거) |
| RUL_clipped 최대 | 125 | PASS |
| engine1 첫 행 RUL / RUL_clipped | 191 / 125 | PASS |
| 결측치(train/test) | 없음 | PASS |

## 7. 다음 단계 (Step 04 모델링, 이 단계에서는 미수행)

- 타깃: `RUL_clipped`를 기본 학습 타깃으로 사용(원본 RUL과 비교 가능하게 둘 다 보존)
- 베이스라인 회귀 → 트리 기반/시계열 모델로 확장
- 평가: RMSE, NASA Score, 그리고 RUL≤30 조기경보의 Recall/FNR (FN 최소화 관점)
- test는 엔진별 마지막 cycle에서 예측해 `RUL_FD001.txt`와 비교

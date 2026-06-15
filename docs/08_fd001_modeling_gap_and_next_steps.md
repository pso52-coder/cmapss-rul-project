# FD001 모델링 한계와 후속 고도화 계획

## 1. 현재 FD001 결과의 위치

현재 프로젝트는 FD001 최적 모델 탐색 완료본이라기보다,
RUL 예측 파이프라인을 완주한 baseline reference이다.

## 2. 현재까지 수행한 모델링

- Baseline mean
- Linear Regression / Ridge
- RandomForest
- GradientBoosting
- RandomForest + 시계열 피처
- threshold별 조기경보 분석

## 3. 부족한 점

- XGBoost / LightGBM 미비교
- validation set 없이 test 결과 기반 threshold 후보 도출
- GroupKFold 등 엔진 단위 검증 미수행
- RUL clipping cap 비교 미수행
- 고장 임박 구간 sample weight 미적용
- 최종 모델 저장 및 재사용 구조 부족

## 4. 후속 실험 계획

| 실험 | 목적 |
|---|---|
| XGBoost / LightGBM 비교 | RF가 최선인지 확인 |
| RUL clipping cap 비교 | label 전략 영향 확인 |
| sample weight 적용 | RUL≤30 구간 탐지 강화 |
| GroupKFold | 엔진 단위 일반화 검증 |
| validation threshold optimization | threshold 선택 정당화 |
| model artifact 저장 | 재사용 가능한 예측 파이프라인 구축 |

## 5. FD002~FD004 확장 전 선행 조건

FD002~FD004로 확장하기 전에,
FD001에서 최종 모델 선정 기준과 threshold 선택 방식을 먼저 고도화한다.

## 6. 우선순위별 실행 계획

### Phase 1. 검증 구조 보완

1. train 데이터를 engine_id 기준으로 train/validation으로 분리한다.
2. validation set에서 threshold 후보를 탐색한다.
3. test set은 최종 평가에만 사용한다.

### Phase 2. 모델 비교

동일한 feature set을 기준으로 다음 모델을 비교한다.

| 모델 | 목적 |
|---|---|
| Ridge / Linear Regression | 단순 기준선 |
| RandomForest | 현재 기준 모델 |
| GradientBoosting | 기존 비교 모델 |
| XGBoost | 강한 boosting 후보 |
| LightGBM | 빠르고 강한 tabular 후보 |

### Phase 3. 고장 임박 구간 강화

- RUL≤30 구간에 sample weight를 부여한다.
- RMSE뿐 아니라 Recall, FNR, Precision, Alert Rate를 함께 비교한다.
- 최종 모델은 RMSE 최소가 아니라 운영 기준을 함께 만족하는 조합으로 선택한다.

### Phase 4. FD002~FD004 확장

FD001에서 최종 모델 선정 기준과 threshold 최적화 방식을 확정한 뒤,
동일한 실험 프레임워크를 FD002~FD004에 적용한다.

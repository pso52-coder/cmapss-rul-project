# FD001 Reference Workflow — C-MAPSS RUL 예측 표준 진행 양식

> FD001에서 완주한 흐름을 FD002~FD004에 재사용하기 위한 표준 템플릿. 각 FD에서 이 양식을 반복하되, **FD별로 다시 판단해야 하는 항목**은 별도 표시한다.

## 0. 진행 원칙

- 순서: 문제정의 → EDA → 전처리 → 모델링 → 평가 → 개선 → 문서화. 건너뛰지 않는다.
- 각 단계: Claude 산출물 → 자체 검증요약 → GPT 검토요청 → GPT 검토 → 다음 단계.
- 명명: `stepNN_Claude_검증요약.txt`, `stepNN_GPT_검토요청.md`, `stepNN_GPT_검토.md`, `cmapss-rul-project_stepNN.zip`.
- 예지보전 핵심: FN(고장 임박 미탐지) 비용 ≫ FP(과잉 정비). RMSE만 보지 않는다.

## 1. 문제 정의 템플릿

- 비즈니스 목표: 누가/무엇을/왜.
- ML 목표: 입력(센서·운전조건 시계열), 출력(RUL 회귀 + RUL≤30 조기경보 분류).
- 성공 기준: RMSE 임계, Recall, FNR 수치.
- 오판정 비용: FN ≫ FP 명시.

## 2. 데이터 로딩 체크리스트

- [ ] 헤더 없음 → 컬럼명 26개 직접 지정 (engine_id, cycle, op_setting_1~3, sensor_1~21).
- [ ] **`sep=r'\s+'`** 사용 (행 끝 공백 2개 함정 회피, NaN 컬럼 생기지 않음).
- [ ] shape·엔진 수·NaN 검증.

## 3. RUL 생성

- train: `RUL = engine별 max_cycle − 현재 cycle`.
- 검증: 한 엔진의 첫 행 RUL = (그 엔진 max_cycle − 1), 마지막 행 = 0.
- 원본 RUL 보존 + Piecewise Linear `RUL_clipped` 별도 생성.

## 4. EDA 체크 항목

- [ ] 엔진별 수명(max_cycle) 분포.
- [ ] RUL 분포(초기 구간 과대 RUL 확인 → 클리핑 근거).
- [ ] 센서 기초통계(스케일 차이 → 정규화 근거).
- [ ] 센서 분산 → **무변화 센서 식별 (FD별 재판단!)**.
- [ ] 대표 엔진 시계열(수명 short/mid/long 구간별).
- [ ] RUL-센서 상관(참고 지표로만, 한계 명시).

## 5. 전처리 결정 기준

- 무변화 센서 제거: 완전상수(std=0)와 준상수(0<std<0.01) 구분 기록. **목록은 FD별 EDA로 재결정.**
- op_setting: **FD001은 단일조건이라 제외. FD002~004는 제거 금지 — 다시 검토.**
- 정규화: MinMax(또는 Z-score). **train으로만 fit, test는 transform만** (누수 방지).
- RUL 클리핑: cap은 최소 엔진 수명보다 낮게(FD001=125). **FD별 수명 분포로 재결정.**

## 6. 모델링 기준

- **baseline부터**: 평균 예측 / cycle 단순 선형회귀.
- 기본 회귀: Linear, Ridge, RandomForest, GradientBoosting. (딥러닝은 후순위)
- 학습 타깃: RUL_clipped.
- **test 평가: 엔진별 마지막 cycle 행만** 추출 → RUL_FDxxx.txt true RUL과 비교. 예측은 [0, cap] 클리핑.

## 7. 평가 기준

- 회귀: RMSE, MAE, NASA Score(늦은 예측에 큰 패널티).
- 조기경보: 실제 위험 = true RUL ≤ 30, 예측 위험 = pred RUL ≤ threshold.
- Precision/Recall/F1/FNR/FP/FN, threshold별(30/40/50/60/65/70) 변화표.
- FN 엔진 분석(ID, true/pred, 오차, 수명구간), 위험구간 평균 잔차.

## 8. 개선 실험 기준

- Step 05에서 진단한 **약점을 겨냥**(예: 위험구간 과대예측 → 시계열 피처).
- 시계열 피처: 이동평균(5/10), 기울기(5/10), 초기 대비 변화량, 현재값−ma5.
- **데이터 누수 방지**: 각 row는 과거+현재 cycle만. 자동 점검(엔진 끝 cycle 제거 후 재계산 일치).
- 기본 모델과 비교(개선 미미해도 과장 금지).
- threshold는 FP 비용 감안한 **운영 정책**으로 남김(확정값 아님).

## 9. 각 단계 산출물 목록

| Step | 코드 | 노트북 | 리포트/문서 |
|---|---|---|---|
| 01 | data_loader, make_rul_dataset | — | docs/01, train_with_rul.csv |
| 02 | — | 01_eda | docs/02 |
| 03 | preprocess | — | docs/03, train/test_preprocessed.csv |
| 04 | modeling | 02_modeling | docs/04, model_results.csv |
| 05 | evaluation_analysis | 03_evaluation_analysis | docs/05, threshold/fn csv |
| 06 | feature_engineering, improvement_modeling | 04_improvement_experiment | docs/06, improvement/threshold csv |
| 07 | — | — | README, docs/07, reference_workflow, extension_plan, final_model_summary.csv |

# STEP_INDEX — cmapss-rul-project 단계별 진행 인덱스

프로젝트 진행 원칙: **문제정의 → 데이터이해(EDA) → 전처리 → 모델링 → 평가**. 단계를 건너뛰지 않고, 각 단계마다 Claude 검증요약 → GPT 검토요청 → GPT 검토를 거친다.

## 현재 기준 진행 상태

- Step 01 완료
- Step 02 완료
- Step 03 완료
- Step 04 완료
- Step 05 완료
- Step 06 완료
- Step 07 완료 (FD001 레퍼런스 문서화) — GPT 검토 대기
- **FD001은 레퍼런스 프로젝트로 정리됨. 다음: FD002~004 확장(별도 진행)**

## 단계 요약

| Step | 단계 | 내용 | 주요 산출물 | 상태 |
|---|---|---|---|---|
| 01 | 세팅 + RUL 생성 | 프로젝트 구조, 문제정의서, 데이터 로딩(sep=r'\s+', 26컬럼), RUL 생성(engine1 첫행=191) | src/data_loader.py, src/make_rul_dataset.py, docs/01_problem_definition.md, data/processed/train_with_rul.csv | 완료 |
| 02 | EDA | 수명/RUL 분포, 센서 분산, 무변화 센서 식별, RUL 상관, 수명 3구간 시계열 | notebooks/01_eda.ipynb, docs/02_eda_summary.md | 완료 |
| 03 | 전처리 | 무변화 센서 7개 제거, op_setting 제외, MinMax 정규화(train fit), Piecewise Linear RUL(cap=125) | src/preprocess.py, data/processed/train_preprocessed.csv, test_preprocessed.csv, docs/03_preprocessing_summary.md | 완료 |
| 04 | 모델링 | Baseline부터 기본 회귀 비교. RMSE 기준 RF/GB 우수했으나 조기경보 FNR 목표 미달. | src/modeling.py, notebooks/02_modeling.ipynb, reports/model_results.csv, docs/04_modeling_summary.md | 완료 |
| 05 | 평가 & 개선 | threshold 조정, FN 엔진 분석, 예측 오차 분석으로 FNR 개선 방향 도출 | notebooks/03_evaluation_analysis.ipynb, docs/05_evaluation_summary.md, reports/threshold_analysis.csv, reports/fn_engine_analysis.csv | 완료 |
| 06 | 개선 실험 | 시계열 피처(이동평균·기울기 등)로 위험구간 과대예측·short-life FN 완화. RF+threshold=40에서 FNR 0.00·Recall 1.00 달성(test 기준). | src/feature_engineering.py, src/improvement_modeling.py, notebooks/04_improvement_experiment.ipynb, reports/improvement_results.csv, reports/threshold_analysis_improved.csv, docs/06_improvement_summary.md | 완료 |
| 07 | 최종 문서화 | FD001 레퍼런스 정리. README 최종화, 최종 요약·표준 워크플로·FD002~004 확장 계획 작성. | README.md, docs/07_final_summary.md, docs/FD001_reference_workflow.md, docs/FD002_003_004_extension_plan.md, reports/final_model_summary.csv | 완료 |

## 명명 규칙

- `stepNN_Claude_검증요약.txt` — Claude 산출물 검증
- `stepNN_GPT_검토요청.md` — GPT에 제출할 검토 요청
- `stepNN_GPT_검토.md` — GPT가 돌려준 검토
- `cmapss-rul-project_stepNN.zip` — 단계별 스냅샷

## 주의사항 (누적)

- 데이터 로딩은 반드시 `sep=r'\s+'` (행 끝 공백 함정 회피, 26컬럼).
- train RUL = engine별 max_cycle − cycle. engine 1 첫 행 RUL = 191 검증.
- 정규화는 train으로만 scaler fit, test는 transform만 (데이터 누수 방지).
- op_setting 제외는 FD001(단일 운전조건) 전용 결정. FD002~004 확장 시 재검토.
- 학습 타깃은 RUL_clipped(cap=125). 평가 시 clipped 기준과 원본 RUL 해석을 구분.
- Step 04 결과는 RMSE 기준 성공이나 조기경보 FNR 기준 목표 미달.
- Step 05는 모델 추가보다 threshold 조정, FN 엔진 분석, 예측 오차 분석을 우선.
- 조기경보 기준: 실제 위험 `true RUL ≤ 30`, 예측 위험 `predicted RUL ≤ threshold`.
- 예지보전에서 FN(고장 임박 미탐지) 비용이 FP(과잉 정비)보다 크다. 과장 금지.
- Step 06 threshold=40은 FD001 test 기준 후보 운영값이며, FD002~004에 그대로 적용하지 않는다.
- 시계열 피처 확장으로 feature 98개 → 과적합 가능성, 엔진 단위 검증은 향후 과제.
- **FD001은 레퍼런스 프로젝트로 정리 완료.** FD002~004는 op_setting·센서제거·정규화·cap·threshold를 FD별 EDA로 재결정하며 Step 01~06을 반복한다.

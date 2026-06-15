"""
src/improvement_modeling.py
----------------------------
Step 06 — 시계열 피처로 모델을 재학습하고 Step 04(센서 현재값만) 결과와 비교한다.
목표: 위험 구간 과대 예측 완화 + short-life 엔진 FN 감소 + FNR 개선.

평가 체계는 Step 04/05와 동일(test = 엔진별 마지막 cycle, true RUL 비교, 예측 [0,125] 클리핑).
threshold는 30/40/50/60/65/70까지 확장(GPT 보완 4.1).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from .feature_engineering import add_timeseries_features, get_engineered_feature_columns
    from .modeling import load_modeling_data, get_test_last_cycle, evaluate, nasa_score
    from .preprocess import get_feature_sensors, RUL_CAP
    from .evaluation_analysis import threshold_table, fn_engine_table, residual_summary, classify_life, ACTUAL_RISK
except ImportError:
    from feature_engineering import add_timeseries_features, get_engineered_feature_columns
    from modeling import load_modeling_data, get_test_last_cycle, evaluate, nasa_score
    from preprocess import get_feature_sensors, RUL_CAP
    from evaluation_analysis import threshold_table, fn_engine_table, residual_summary, classify_life, ACTUAL_RISK

RANDOM_STATE = 42


def build_feature_datasets(root):
    """train/test에 시계열 피처를 만들고, RUL_clipped 타깃과 함께 반환."""
    root = Path(root)
    proc = root / "data" / "processed"
    train_pp = pd.read_csv(proc / "train_preprocessed.csv")
    test_pp = pd.read_csv(proc / "test_preprocessed.csv")

    train_fe = add_timeseries_features(train_pp)
    test_fe = add_timeseries_features(test_pp)
    feat_cols = get_engineered_feature_columns()
    return train_fe, test_fe, feat_cols


def run_improved_models(root):
    """시계열 피처로 RF/GB/Ridge 재학습 후 test(엔진별 마지막행) 평가."""
    root = Path(root)
    train_fe, test_fe, feat_cols = build_feature_datasets(root)
    _, _, true_rul = load_modeling_data(root)

    X_train = train_fe[feat_cols].values
    y_train = train_fe["RUL_clipped"].values

    test_last = get_test_last_cycle(test_fe)
    X_test = test_last[feat_cols].values
    y_test_true = true_rul.loc[test_last["engine_id"].values, "RUL"].values

    models = {
        "Ridge_ts": Ridge(alpha=1.0),
        "RandomForest_ts": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting_ts": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    results, preds = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        results[name] = evaluate(y_test_true, p)
        preds[name] = np.clip(p, 0, RUL_CAP)

    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    return results_df, preds, y_test_true, test_last, train_fe


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    results_df, preds, y_true, test_last, train_fe = run_improved_models(root)

    print("=== 시계열 피처 모델 성능 (test, true RUL 기준) ===")
    print(results_df.to_string())

    # Step 04 결과와 비교
    step04 = pd.read_csv(root / "reports" / "model_results.csv", index_col=0)
    print("\n=== Step 04(현재값) vs Step 06(시계열) 핵심 비교 ===")
    cmp = pd.DataFrame({
        "RMSE_step04": [step04.loc["RandomForest", "RMSE"], step04.loc["GradientBoosting", "RMSE"]],
        "RMSE_step06": [results_df.loc["RandomForest_ts", "RMSE"], results_df.loc["GradientBoosting_ts", "RMSE"]],
        "FNR_step04": [step04.loc["RandomForest", "FNR"], step04.loc["GradientBoosting", "FNR"]],
        "FNR_step06": [results_df.loc["RandomForest_ts", "FNR"], results_df.loc["GradientBoosting_ts", "FNR"]],
        "Recall_step04": [step04.loc["RandomForest", "Alert_Recall"], step04.loc["GradientBoosting", "Alert_Recall"]],
        "Recall_step06": [results_df.loc["RandomForest_ts", "Alert_Recall"], results_df.loc["GradientBoosting_ts", "Alert_Recall"]],
    }, index=["RandomForest", "GradientBoosting"])
    print(cmp.to_string())

    # threshold 확장 분석 (30~70)
    thr = (30, 40, 50, 60, 65, 70)
    print("\n=== threshold 확장: RandomForest_ts ===")
    t_rf = threshold_table(y_true, preds["RandomForest_ts"], thresholds=thr)
    print(t_rf.to_string(index=False))

    # 위험구간 잔차 비교
    print("\n=== 위험구간(RUL<=30) 잔차: RandomForest_ts ===")
    rs = residual_summary(y_true, preds["RandomForest_ts"])
    for k, v in rs.items():
        print(f"  {k}: {v}")

    # FN 엔진
    fn_rf = fn_engine_table(test_last, y_true, preds["RandomForest_ts"], train_fe)
    print(f"\n=== FN 엔진 (RandomForest_ts, threshold=30): {len(fn_rf)}대 ===")
    if len(fn_rf):
        print(fn_rf.to_string(index=False))

    # 저장
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    results_df.to_csv(reports / "improvement_results.csv")
    t_rf.assign(model="RandomForest_ts").to_csv(reports / "threshold_analysis_improved.csv", index=False)
    print(f"\n저장: improvement_results.csv, threshold_analysis_improved.csv")

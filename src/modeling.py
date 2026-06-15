"""
src/modeling.py
----------------
Step 04 모델링 — 전처리 데이터셋으로 baseline부터 기본 회귀 모델까지 학습/평가한다.

핵심 원칙 (GPT 검토 반영):
1. Baseline부터 시작한다 (평균 예측 / cycle 단순 선형회귀).
2. test 평가는 전체 행이 아니라 **engine_id별 마지막 cycle 행**만 사용하고,
   RUL_FD001.txt의 true RUL과 결합해 비교한다.
3. RMSE/MAE뿐 아니라 RUL≤30 조기경보 Recall / False Negative Rate를 함께 본다.
   (이 프로젝트는 고장 임박을 놓치는 FN이 가장 위험하다.)

학습 타깃: RUL_clipped (cap=125). 단, test 평가의 true RUL은 원본(클리핑 안 됨)이므로
예측값도 동일 비교를 위해 [0, 125]로 클리핑한 뒤 비교한다(타깃 정의와 평가 기준 일치).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from .preprocess import get_feature_sensors, RUL_CAP
except ImportError:
    from preprocess import get_feature_sensors, RUL_CAP

ALERT_THRESHOLD = 30  # RUL <= 30 이면 "고장 임박" 경보


# ─────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────
def load_modeling_data(root):
    """train_preprocessed, test_preprocessed, RUL_FD001(true RUL) 로딩."""
    root = Path(root)
    proc = root / "data" / "processed"
    train = pd.read_csv(proc / "train_preprocessed.csv")
    test = pd.read_csv(proc / "test_preprocessed.csv")
    true_rul = pd.read_csv(root / "data" / "raw" / "RUL_FD001.txt",
                           sep=r"\s+", header=None, names=["RUL"])
    true_rul.index = np.arange(1, len(true_rul) + 1)  # engine_id 1..100
    return train, test, true_rul


def get_test_last_cycle(test):
    """test에서 engine_id별 마지막 cycle 행만 추출 (평가 시점)."""
    idx = test.groupby("engine_id")["cycle"].idxmax()
    return test.loc[idx].sort_values("engine_id").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 평가 지표
# ─────────────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    """NASA C-MAPSS scoring: 늦은 예측(과대, d>0)에 더 큰 패널티.
    d = y_pred - y_true.  d<0 -> exp(-d/13)-1,  d>=0 -> exp(d/10)-1."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred, clip_pred=True):
    """RMSE/MAE/NASA Score + RUL<=30 조기경보 Recall/FNR.

    y_true: 원본 true RUL (test 정답)
    y_pred: 모델 예측. clip_pred=True면 [0, RUL_CAP]로 클리핑 (학습 타깃과 기준 일치)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if clip_pred:
        y_pred = np.clip(y_pred, 0, RUL_CAP)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    score = nasa_score(y_true, y_pred)

    # 조기경보: 실제 고장임박(RUL<=30)을 예측이 얼마나 잡는가
    actual_alert = y_true <= ALERT_THRESHOLD
    pred_alert = y_pred <= ALERT_THRESHOLD
    tp = int(np.sum(actual_alert & pred_alert))
    fn = int(np.sum(actual_alert & ~pred_alert))  # 실제 임박인데 정상으로 예측 = 놓침(위험)
    n_alert = int(np.sum(actual_alert))
    recall = (tp / n_alert) if n_alert > 0 else float("nan")
    fnr = (fn / n_alert) if n_alert > 0 else float("nan")

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "NASA_Score": round(score, 1),
        "Alert_Recall": round(recall, 3),
        "FNR": round(fnr, 3),
        "n_actual_alert": n_alert,
        "FN_count": fn,
    }


# ─────────────────────────────────────────────────────────────
# 모델 학습/평가 파이프라인
# ─────────────────────────────────────────────────────────────
def run_all_models(root, random_state=42):
    """baseline 2종 + 회귀 4종을 학습하고 test(엔진별 마지막행)로 평가."""
    train, test, true_rul = load_modeling_data(root)
    feats = get_feature_sensors()

    X_train = train[feats].values
    y_train = train["RUL_clipped"].values
    cycle_train = train[["cycle"]].values

    test_last = get_test_last_cycle(test)
    X_test = test_last[feats].values
    cycle_test = test_last[["cycle"]].values
    y_test_true = true_rul.loc[test_last["engine_id"].values, "RUL"].values  # 원본 true RUL

    results = {}
    preds = {}

    # Baseline 1: train RUL_clipped 평균 예측
    mean_pred = np.full(len(y_test_true), y_train.mean())
    results["Baseline_mean"] = evaluate(y_test_true, mean_pred)
    preds["Baseline_mean"] = mean_pred

    # Baseline 2: cycle 기반 단순 선형회귀
    lr_cycle = LinearRegression().fit(cycle_train, y_train)
    pred_cycle = lr_cycle.predict(cycle_test)
    results["Baseline_cycle_LR"] = evaluate(y_test_true, pred_cycle)
    preds["Baseline_cycle_LR"] = np.clip(pred_cycle, 0, RUL_CAP)

    # 회귀 모델 4종 (센서 피처 사용)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        results[name] = evaluate(y_test_true, p)
        preds[name] = np.clip(p, 0, RUL_CAP)

    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    return results_df, preds, y_test_true, test_last


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    results_df, preds, y_true, test_last = run_all_models(root)

    print("=== 모델별 성능 (test: 엔진별 마지막 cycle, true RUL 기준) ===")
    print(results_df.to_string())

    out = root / "reports"
    out.mkdir(exist_ok=True)
    results_df.to_csv(out / "model_results.csv")
    print(f"\n저장: {out / 'model_results.csv'}")

    best = results_df["RMSE"].idxmin()
    print(f"\nRMSE 최저 모델: {best} (RMSE={results_df.loc[best,'RMSE']})")

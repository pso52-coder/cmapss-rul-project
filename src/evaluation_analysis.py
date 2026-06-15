"""
src/evaluation_analysis.py
---------------------------
Step 05 평가/개선 — 현재 모델의 한계를 분석하고 FNR 개선 방향을 도출한다.
(새 모델을 많이 추가하지 않는다. 기존 RF/GB 중심 분석)

분석:
1. threshold별 조기경보 성능 (30/40/50/60)
   - 실제 위험 = true RUL <= 30
   - 예측 위험 = predicted RUL <= threshold
   - Precision / Recall / F1 / FNR / FP / FN
2. False Negative 엔진 분석 (threshold=30 기준)
   - FN 엔진 ID, true/pred RUL, 오차, 수명 구간(short/mid/long)
3. 예측 오차 분석 (잔차, 과대/과소 예측)
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .modeling import run_all_models, load_modeling_data
    from .preprocess import RUL_CAP
except ImportError:
    from modeling import run_all_models, load_modeling_data
    from preprocess import RUL_CAP

ACTUAL_RISK = 30  # 실제 위험 정의: true RUL <= 30 (고정)


def threshold_table(y_true, y_pred, thresholds=(30, 40, 50, 60)):
    """예측 위험 임계값(predicted RUL <= t)별 성능 표."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    actual = y_true <= ACTUAL_RISK
    rows = []
    for t in thresholds:
        pred = y_pred <= t
        tp = int(np.sum(actual & pred))
        fp = int(np.sum(~actual & pred))
        fn = int(np.sum(actual & ~pred))
        tn = int(np.sum(~actual & ~pred))
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and not np.isnan(precision) and not np.isnan(recall) else float("nan"))
        fnr = fn / (tp + fn) if (tp + fn) else float("nan")
        rows.append({
            "threshold": t, "Precision": round(precision, 3), "Recall": round(recall, 3),
            "F1": round(f1, 3), "FNR": round(fnr, 3), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


def classify_life(total_life, train_life):
    """train 수명 분포의 33/66 분위로 short/mid/long 구간 분류."""
    q1, q2 = np.percentile(train_life, [33, 66])
    if total_life <= q1:
        return "short"
    if total_life <= q2:
        return "mid"
    return "long"


def fn_engine_table(test_last, y_true, y_pred, train, threshold=ACTUAL_RISK):
    """threshold(기본 30) 기준 False Negative 엔진 분석 표."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    actual = y_true <= ACTUAL_RISK
    pred_alert = y_pred <= threshold
    fn_mask = actual & ~pred_alert

    train_life = train.groupby("engine_id")["cycle"].max().values
    rows = []
    for i, is_fn in enumerate(fn_mask):
        if not is_fn:
            continue
        last_cycle = int(test_last.iloc[i]["cycle"])
        true_rul = int(y_true[i])
        total_life = last_cycle + true_rul  # 관측 마지막 + 잔여 = 추정 총 수명
        rows.append({
            "engine_id": int(test_last.iloc[i]["engine_id"]),
            "true_RUL": true_rul,
            "pred_RUL": round(float(y_pred[i]), 1),
            "error(pred-true)": round(float(y_pred[i] - true_rul), 1),
            "last_obs_cycle": last_cycle,
            "est_total_life": total_life,
            "life_group": classify_life(total_life, train_life),
        })
    return pd.DataFrame(rows)


def residual_summary(y_true, y_pred):
    """잔차(예측-실제) 요약 + 고장임박 구간 별도."""
    y_true = np.asarray(y_true); y_pred = np.clip(np.asarray(y_pred), 0, RUL_CAP)
    resid = y_pred - y_true
    alert = y_true <= ACTUAL_RISK
    return {
        "n": len(y_true),
        "mean_residual": round(float(resid.mean()), 2),
        "over_pred_count": int(np.sum(resid > 0)),   # 과대 예측(위험: 실제보다 길게 봄)
        "under_pred_count": int(np.sum(resid < 0)),  # 과소 예측(보수적)
        "alert_zone_mean_residual": round(float(resid[alert].mean()), 2) if alert.any() else float("nan"),
        "alert_zone_over_pred": int(np.sum(resid[alert] > 0)) if alert.any() else 0,
        "alert_zone_n": int(alert.sum()),
    }


def run_evaluation_analysis(root):
    results_df, preds, y_true, test_last = run_all_models(root)
    train, _, _ = load_modeling_data(root)

    out = {"results_df": results_df, "preds": preds, "y_true": y_true, "test_last": test_last}
    for m in ["RandomForest", "GradientBoosting"]:
        out[f"threshold_{m}"] = threshold_table(y_true, preds[m])
        out[f"fn_{m}"] = fn_engine_table(test_last, y_true, preds[m], train)
        out[f"resid_{m}"] = residual_summary(y_true, preds[m])
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    r = run_evaluation_analysis(root)

    print("=== threshold별 성능: RandomForest ===")
    print(r["threshold_RandomForest"].to_string(index=False))
    print("\n=== threshold별 성능: GradientBoosting ===")
    print(r["threshold_GradientBoosting"].to_string(index=False))
    print("\n=== FN 엔진 (RandomForest, threshold=30) ===")
    print(r["fn_RandomForest"].to_string(index=False))
    print("\n=== 잔차 요약 (RandomForest) ===")
    for k, v in r["resid_RandomForest"].items():
        print(f"  {k}: {v}")

    # 저장
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    # threshold 표는 두 모델 합쳐 저장
    t_rf = r["threshold_RandomForest"].assign(model="RandomForest")
    t_gb = r["threshold_GradientBoosting"].assign(model="GradientBoosting")
    pd.concat([t_rf, t_gb]).to_csv(reports / "threshold_analysis.csv", index=False)
    r["fn_RandomForest"].assign(model="RandomForest").to_csv(reports / "fn_engine_analysis.csv", index=False)
    print(f"\n저장: {reports/'threshold_analysis.csv'}, {reports/'fn_engine_analysis.csv'}")

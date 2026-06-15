"""
src/feature_engineering.py
---------------------------
Step 06 — 시계열 피처 생성. Step 05에서 확인한 약점(위험 구간 과대 예측,
short-life 엔진 FN)을 겨냥해 "열화 속도"를 모델이 보게 한다.

데이터 누수 방지 (핵심):
- 각 row의 피처는 그 row의 cycle 이전+현재 값만 사용한다.
- rolling/diff 모두 engine_id별로 시간 순서를 지켜 계산하며, 미래 cycle 값을 쓰지 않는다.
- test도 동일 방식(엔진별 시간순)으로 생성한다.
- 정규화된 train/test_preprocessed를 입력으로 받으므로, 스케일러 누수도 없다(Step 03에서 train fit).

생성 피처 (유효 센서 14개 각각에 대해):
- {s}_ma5, {s}_ma10   : 최근 5/10 cycle 이동평균 (현재 포함, 과거만)
- {s}_slope5, {s}_slope10 : 최근 5/10 cycle 선형 기울기 (열화 속도)
- {s}_delta_init      : 초기값(첫 cycle) 대비 변화량
- {s}_dev_ma5         : 현재값 - ma5 (최근 평균에서의 이탈)
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .preprocess import get_feature_sensors
except ImportError:
    from preprocess import get_feature_sensors


def _rolling_slope(series, window):
    """최근 window개 값의 선형 기울기(시간당 변화율). 과거+현재만 사용.
    값이 window개 미만이면 가능한 만큼으로 계산, 1개면 0."""
    x = np.arange(len(series))
    def slope(arr):
        n = len(arr)
        if n < 2:
            return 0.0
        xi = np.arange(n)
        # polyfit degree 1
        return float(np.polyfit(xi, arr, 1)[0])
    return series.rolling(window, min_periods=2).apply(slope, raw=True).fillna(0.0)


def add_timeseries_features(df, sensors=None):
    """engine_id별 시간순으로 시계열 피처를 추가한다 (미래값 미사용)."""
    if sensors is None:
        sensors = get_feature_sensors()
    df = df.sort_values(["engine_id", "cycle"]).copy()
    g = df.groupby("engine_id")

    new_cols = {}
    for s in sensors:
        # 이동평균 (현재 포함, 과거만)
        new_cols[f"{s}_ma5"] = g[s].transform(lambda x: x.rolling(5, min_periods=1).mean())
        new_cols[f"{s}_ma10"] = g[s].transform(lambda x: x.rolling(10, min_periods=1).mean())
        # 기울기 (열화 속도)
        new_cols[f"{s}_slope5"] = g[s].transform(lambda x: _rolling_slope(x, 5))
        new_cols[f"{s}_slope10"] = g[s].transform(lambda x: _rolling_slope(x, 10))
        # 초기값 대비 변화량
        new_cols[f"{s}_delta_init"] = df[s] - g[s].transform("first")
        # 현재값 - 최근 이동평균
        new_cols[f"{s}_dev_ma5"] = df[s] - new_cols[f"{s}_ma5"]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def get_engineered_feature_columns(sensors=None):
    """추가된 시계열 피처 컬럼명 목록 (원본 센서 + 파생)."""
    if sensors is None:
        sensors = get_feature_sensors()
    suffixes = ["", "_ma5", "_ma10", "_slope5", "_slope10", "_delta_init", "_dev_ma5"]
    cols = []
    for s in sensors:
        for suf in suffixes:
            cols.append(f"{s}{suf}")
    return cols


def validate_no_leakage(df_raw, sensors=None):
    """누수 점검: 한 엔진의 특정 cycle 피처가 그 이후 값에 의존하지 않는지 확인.
    방법: 엔진을 마지막 cycle 한 칸 잘라낸 부분집합으로 피처를 다시 만들어,
    겹치는 cycle 구간의 피처가 동일한지 비교(미래 미사용이면 동일해야 함)."""
    if sensors is None:
        sensors = get_feature_sensors()
    eid = df_raw["engine_id"].iloc[0]
    sub = df_raw[df_raw["engine_id"] == eid].sort_values("cycle")
    full = add_timeseries_features(sub, sensors)
    truncated = add_timeseries_features(sub.iloc[:-1], sensors)  # 마지막 cycle 제거
    cols = get_engineered_feature_columns(sensors)
    # 겹치는 행(=truncated 전체)에서 피처가 같아야 함
    a = full.iloc[:-1][cols].reset_index(drop=True)
    b = truncated[cols].reset_index(drop=True)
    max_diff = float((a - b).abs().max().max())
    return {"engine_checked": int(eid), "max_feature_diff": max_diff, "leak_free": max_diff < 1e-9}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    proc = root / "data" / "processed"
    train = pd.read_csv(proc / "train_preprocessed.csv")
    test = pd.read_csv(proc / "test_preprocessed.csv")

    # 누수 점검
    leak = validate_no_leakage(train)
    print("=== 누수 점검 ===")
    for k, v in leak.items():
        print(f"  {k}: {v}")

    train_fe = add_timeseries_features(train)
    test_fe = add_timeseries_features(test)
    feat_cols = get_engineered_feature_columns()
    print(f"\n원본 센서 14개 → 시계열 피처 포함 {len(feat_cols)}개")
    print("train_fe shape:", train_fe.shape, "| NaN:", bool(train_fe[feat_cols].isna().any().any()))
    print("test_fe shape :", test_fe.shape, "| NaN:", bool(test_fe[feat_cols].isna().any().any()))

    train_fe.to_csv(proc / "train_features.csv", index=False)
    test_fe.to_csv(proc / "test_features.csv", index=False)
    print(f"\n저장: {proc/'train_features.csv'}, {proc/'test_features.csv'}")

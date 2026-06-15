"""
src/preprocess.py
------------------
Step 03 전처리 — 모델링에 사용할 학습용 데이터셋을 생성한다. (모델링은 하지 않음)

수행 작업:
1. 무변화 센서 제거
   - 완전 상수(std=0): sensor_1, 10, 18, 19  → 제거 확정
   - 준상수(0<std<0.01): sensor_5, 6, 16     → 제거 후보. 기본 전처리에서는 함께 제거
2. 운전조건 처리
   - FD001은 단일 운전 조건이라 op_setting_1~3이 거의 상수 → 피처에서 제외
3. 센서 정규화 (Min-Max)
   - train 기준으로만 scaler를 fit, test에는 transform만 적용 → 데이터 누수 방지
4. Piecewise Linear RUL 클리핑
   - cap=125 (C-MAPSS 문헌 표준, 최소 엔진 수명 128보다 낮아 전 구간 클리핑 엔진 없음)
   - 원본 RUL과 RUL_clipped를 둘 다 저장

왜 이렇게 하는가:
- 정보 없는 센서를 빼면 모델이 노이즈를 학습하지 않는다.
- 센서 스케일이 제각각이라 정규화 없이는 일부 센서가 과도한 영향을 준다.
- 선형 RUL은 초기 구간을 비현실적으로 크게 라벨링하므로, 일정 값 이상을 상한으로
  고정(Piecewise Linear)해 "열화가 시작된 뒤"의 신호에 집중하게 한다.
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# EDA에서 도출된 무변화 센서
CONSTANT_SENSORS = ["sensor_1", "sensor_10", "sensor_18", "sensor_19"]  # std=0, 제거 확정
NEAR_CONSTANT_SENSORS = ["sensor_5", "sensor_6", "sensor_16"]            # 0<std<0.01, 제거 후보
DROP_SENSORS = CONSTANT_SENSORS + NEAR_CONSTANT_SENSORS                  # 기본 전처리: 7개 제거
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]          # FD001 단일조건 → 제외

RUL_CAP = 125  # Piecewise Linear RUL 클리핑 상한


def get_feature_sensors():
    """제거 후 남는 유효 센서 14개 목록을 반환한다."""
    all_sensors = [f"sensor_{i}" for i in range(1, 22)]
    return [s for s in all_sensors if s not in DROP_SENSORS]


def add_train_rul(train):
    """train에 RUL = engine별 max_cycle - cycle 컬럼을 추가한다."""
    df = train.copy()
    df["RUL"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    return df


def clip_rul(df, cap=RUL_CAP):
    """원본 RUL은 보존하고 RUL_clipped(상한 cap)를 추가한다."""
    df = df.copy()
    df["RUL_clipped"] = df["RUL"].clip(upper=cap)
    return df


def preprocess(train_raw, test_raw, cap=RUL_CAP):
    """train/test를 전처리한다.

    - 무변화 센서 7개 제거, op_setting 제외
    - train 기준 MinMax fit → train/test 모두 transform (누수 방지)
    - train에 RUL, RUL_clipped 생성

    Returns
    -------
    train_out : DataFrame  (engine_id, cycle, 정규화 센서14, RUL, RUL_clipped)
    test_out  : DataFrame  (engine_id, cycle, 정규화 센서14)  ※ RUL은 RUL_FD001.txt로 별도 평가
    scaler    : 학습된 MinMaxScaler
    feature_sensors : list
    """
    feats = get_feature_sensors()

    # RUL 생성 (train만)
    train = add_train_rul(train_raw)
    train = clip_rul(train, cap=cap)

    # 정규화: train으로만 fit
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test_raw.copy()
    train_scaled[feats] = scaler.fit_transform(train[feats])
    test_scaled[feats] = scaler.transform(test_raw[feats])

    # 필요한 컬럼만 선택 (op_setting, 제거 센서 제외)
    train_out = train_scaled[["engine_id", "cycle"] + feats + ["RUL", "RUL_clipped"]]
    test_out = test_scaled[["engine_id", "cycle"] + feats]
    return train_out, test_out, scaler, feats


def validate_preprocessed(train_out, test_out, feats):
    """전처리 결과 검증."""
    res = {
        "train_shape": train_out.shape,
        "test_shape": test_out.shape,
        "n_feature_sensors": len(feats),
        "dropped_sensors": DROP_SENSORS,
        "rul_clip_max": int(train_out["RUL_clipped"].max()),
        "rul_original_max": int(train_out["RUL"].max()),
        # 정규화 범위: train의 각 피처가 [0,1]에 들어오는지
        "train_feat_min": float(train_out[feats].min().min()),
        "train_feat_max": float(train_out[feats].max().max()),
        "train_has_nan": bool(train_out.isna().any().any()),
        "test_has_nan": bool(test_out.isna().any().any()),
        "engine1_first_rul": int(train_out[train_out.engine_id == 1].iloc[0]["RUL"]),
        "engine1_first_rul_clipped": int(train_out[train_out.engine_id == 1].iloc[0]["RUL_clipped"]),
    }
    res["passed"] = (
        len(feats) == 14
        and res["rul_clip_max"] == RUL_CAP
        and abs(res["train_feat_min"] - 0.0) < 1e-9
        and abs(res["train_feat_max"] - 1.0) < 1e-9
        and not res["train_has_nan"]
        and not res["test_has_nan"]
        and res["engine1_first_rul"] == 191
        and res["engine1_first_rul_clipped"] == RUL_CAP
    )
    return res


if __name__ == "__main__":
    try:
        from .data_loader import load_fd001_data
    except ImportError:
        from data_loader import load_fd001_data

    root = Path(__file__).resolve().parent.parent
    train_raw, test_raw, _ = load_fd001_data(root / "data" / "raw")

    train_out, test_out, scaler, feats = preprocess(train_raw, test_raw)
    res = validate_preprocessed(train_out, test_out, feats)

    print("=== 전처리 검증 ===")
    for k, v in res.items():
        print(f"  {k}: {v}")

    out_dir = root / "data" / "processed"
    train_path = out_dir / "train_preprocessed.csv"
    test_path = out_dir / "test_preprocessed.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"\n저장: {train_path}")
    print(f"저장: {test_path}")
    print("PASS" if res["passed"] else "FAIL")

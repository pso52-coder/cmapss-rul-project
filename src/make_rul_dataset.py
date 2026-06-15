"""
src/make_rul_dataset.py
------------------------
train 데이터에 타깃 변수 RUL(Remaining Useful Life)을 생성하고,
data/processed/train_with_rul.csv 로 저장하는 모듈.

왜 이 작업이 필요한가:
- train은 각 엔진이 고장날 때까지 기록(run-to-failure)되므로,
  엔진별 마지막 사이클이 곧 고장 시점이다.
  따라서 각 시점의 잔여 수명은 RUL = (그 엔진의 max_cycle) - (현재 cycle).
- 이 타깃이 잘못 만들어지면 이후 모든 모델링이 무효가 되므로,
  생성 직후 engine 1의 첫 행/마지막 행 값을 자동 검증한다.
"""

from pathlib import Path
import pandas as pd

# 같은 폴더(src)의 모듈을 단독 실행/임포트 양쪽에서 쓰기 위한 처리
try:
    from .data_loader import load_fd001_data, validate_loaded_data
except ImportError:  # 스크립트로 직접 실행할 때
    from data_loader import load_fd001_data, validate_loaded_data


def add_train_rul(train):
    """train DataFrame에 RUL 컬럼을 추가해 반환한다.

    RUL = engine_id별 max_cycle - 현재 cycle
    (원본을 변형하지 않도록 복사본에 작업)
    """
    df = train.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def validate_train_rul(train_with_rul):
    """RUL 생성 결과를 검증하고 결과 dict를 반환한다.

    핵심 검증:
    - engine 1 첫 행 RUL == 191 (max_cycle 192 - cycle 1)
    - engine 1 마지막 행 RUL == 0 (고장 시점)
    - RUL 최소값 0 / 음수 없음 / 결측 없음
    """
    e1 = train_with_rul[train_with_rul["engine_id"] == 1].sort_values("cycle")
    e1_first = int(e1.iloc[0]["RUL"])
    e1_last = int(e1.iloc[-1]["RUL"])
    rul_min = int(train_with_rul["RUL"].min())
    rul_max = int(train_with_rul["RUL"].max())
    has_nan = bool(train_with_rul.isna().any().any())

    results = {
        "shape": train_with_rul.shape,
        "engine1_first_rul": e1_first,
        "engine1_last_rul": e1_last,
        "rul_min": rul_min,
        "rul_max": rul_max,
        "has_nan": has_nan,
        "no_negative_rul": rul_min >= 0,
    }
    results["passed"] = (
        e1_first == 191
        and e1_last == 0
        and rul_min == 0
        and not has_nan
    )
    return results


def save_processed_train(train_with_rul, output_path="data/processed/train_with_rul.csv"):
    """RUL이 추가된 train을 CSV로 저장한다. (인덱스 제외)"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_with_rul.to_csv(out, index=False)
    return str(out)


if __name__ == "__main__":
    # 1) 로딩 + 로딩 검증
    train, test, rul = load_fd001_data()
    load_res = validate_loaded_data(train, test, rul)
    assert load_res["passed"], f"로딩 검증 실패: {load_res}"

    # 2) RUL 생성 + RUL 검증
    train_with_rul = add_train_rul(train)
    rul_res = validate_train_rul(train_with_rul)
    print("=== RUL 생성 검증 ===")
    for k, v in rul_res.items():
        print(f"  {k}: {v}")
    assert rul_res["passed"], f"RUL 검증 실패: {rul_res}"

    # 3) 저장
    path = save_processed_train(train_with_rul)
    print(f"\n저장 완료: {path}")
    print("PASS" if rul_res["passed"] else "FAIL")

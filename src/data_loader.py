"""
src/data_loader.py
-------------------
NASA C-MAPSS FD001 데이터를 안정적으로 로딩하는 모듈.

왜 이 모듈이 필요한가:
- FD001 원본은 헤더가 없고, 공백으로 구분되며, 행 끝에 공백이 2개 붙어 있다.
  이 함정(컬럼명 부재 + 행 끝 공백)을 코드 한 곳에 가둬, 노트북마다 매번
  같은 실수를 반복하지 않도록 로딩 로직을 함수로 고정한다.
"""

from pathlib import Path
import pandas as pd


def get_fd001_columns():
    """FD001의 26개 컬럼명을 순서대로 반환한다.

    구조: engine_id(1) + cycle(1) + op_setting_1~3(3) + sensor_1~21(21) = 26
    원본에는 헤더가 없으므로 read_csv 시 names 인자로 직접 지정해야 한다.
    """
    base = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    sensors = [f"sensor_{i}" for i in range(1, 22)]
    return base + sensors


def load_fd001_data(raw_dir="data/raw"):
    """FD001의 train / test / RUL 세 파일을 한 번에 로딩한다.

    핵심 포인트:
    - sep=r'\\s+' 사용: sep=' '로 읽으면 행 끝 공백 2개 때문에 NaN 컬럼 2개가
      추가로 생겨 28컬럼이 된다. 정규식 공백 구분자로 읽어야 정확히 26컬럼이 된다.
    - header=None + names=...: 원본에 헤더가 없으므로 컬럼명을 직접 부여한다.

    Returns
    -------
    train : DataFrame  (run-to-failure 완전 이력, 26컬럼)
    test  : DataFrame  (임의 시점 절단, 26컬럼)
    rul   : DataFrame  (test 엔진별 정답 RUL, 1컬럼 'RUL')
    """
    raw = Path(raw_dir)
    cols = get_fd001_columns()

    train = pd.read_csv(raw / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(raw / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(raw / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

    return train, test, rul


def validate_loaded_data(train, test, rul):
    """로딩 결과가 기대한 구조인지 검증하고 결과 dict를 반환한다.

    검증 항목:
    - train/test 컬럼 수 26개
    - train 100엔진 / 20,631행, test 100엔진 / 13,096행
    - RUL 100행
    - NaN 컬럼이 없는지 (sep 함정에 빠지지 않았는지 확인)
    """
    expected_cols = get_fd001_columns()
    results = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "rul_shape": rul.shape,
        "train_columns_ok": list(train.columns) == expected_cols,
        "test_columns_ok": list(test.columns) == expected_cols,
        "train_n_engines": int(train["engine_id"].nunique()),
        "test_n_engines": int(test["engine_id"].nunique()),
        "train_has_nan": bool(train.isna().any().any()),
        "test_has_nan": bool(test.isna().any().any()),
        "rul_has_nan": bool(rul.isna().any().any()),
    }

    # 기대값과 비교한 통과 여부
    results["passed"] = (
        train.shape == (20631, 26)
        and test.shape == (13096, 26)
        and rul.shape == (100, 1)
        and results["train_columns_ok"]
        and results["test_columns_ok"]
        and results["train_n_engines"] == 100
        and results["test_n_engines"] == 100
        and not results["train_has_nan"]
        and not results["test_has_nan"]
    )
    return results


if __name__ == "__main__":
    # 단독 실행 시 로딩 + 검증 결과를 출력한다.
    train, test, rul = load_fd001_data()
    res = validate_loaded_data(train, test, rul)
    print("=== FD001 로딩 검증 ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("PASS" if res["passed"] else "FAIL")

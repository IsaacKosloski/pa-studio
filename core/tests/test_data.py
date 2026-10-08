from pathlib import Path

import numpy as np
import pytest
from core.data import load_iq_csv, split_contiguous


def test_load_iq_csv_builds_complex_input_and_output(tmp_path: Path) -> None:
    csv = tmp_path / "pa.csv"
    csv.write_text("Xreal,Ximg,Yreal,Yimg\n1,2,10,20\n3,4,30,40\n")

    x, y = load_iq_csv(csv)

    np.testing.assert_allclose(x, [1 + 2j, 3 + 4j])
    np.testing.assert_allclose(y, [10 + 20j, 30 + 40j])


def test_load_iq_csv_wrong_column_count_raises(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "pa.csv"
    csv.write_text("Xreal,Ximg,Yreal\n1,2,10\n3,4,30\n")

    with pytest.raises(ValueError, match="4"):
        load_iq_csv(csv)


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(100, dtype=float)
    y = np.arange(100, dtype=float) * 2
    return x, y


def test_split_contiguous_shape_mismatch_raises() -> None:
    x = np.ones(10)
    y = np.ones(15)
    with pytest.raises(ValueError, match="same shape"):
        split_contiguous(x, y)


def test_split_contiguous_default_fractions_sizes(
    dummy_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = dummy_data
    splits = split_contiguous(x, y)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits

    assert len(x_train) == 60
    assert len(x_val) == 20
    assert len(x_test) == 20

    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20


def test_split_contiguous_with_imperfect_division(
    dummy_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = dummy_data
    splits = split_contiguous(x, y, fractions=(1 / 3, 1 / 3, 1 - 2 / 3))

    (x_train, _), (x_val, _), (x_test, _) = splits

    assert len(x_train) == 33
    assert len(x_val) == 33
    assert len(x_test) == 34


def test_split_contiguous_fractions_sum_error() -> None:
    x = np.ones(10)
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        split_contiguous(x, x, fractions=(0.5, 0.2, 0.2))


def test_split_contiguous_large_n_sizes() -> None:
    N = 48384
    x = np.zeros(N)
    y = np.zeros(N)

    splits = split_contiguous(x, y)
    (x_tr, _), (x_va, _), (x_te, _) = splits

    assert len(x_tr) == 29030
    assert len(x_va) == 9676
    assert len(x_te) == 9678


def test_split_contiguous_blocks_rebuild_the_original_signal() -> None:
    x = np.arange(100) + 0j
    y = 2 * x

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split_contiguous(x, y)

    assert (len(x_tr), len(x_va), len(x_te)) == (60, 20, 20)

    np.testing.assert_array_equal(np.concatenate([x_tr, x_va, x_te]), x)
    np.testing.assert_array_equal(np.concatenate([y_tr, y_va, y_te]), y)

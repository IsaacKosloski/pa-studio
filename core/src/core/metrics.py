import numpy as np


def _check_inputs(y: np.ndarray, y_hat: np.ndarray) -> None:
    """Raise ValueError if the shapes differ or y has zero energy."""
    if y.shape != y_hat.shape:
        raise ValueError(
            f"Inputs must have the same shape. Got {y.shape} and {y_hat.shape}"
        )

    if np.sum(np.abs(y) ** 2) == 0:
        raise ValueError("Reference signal y has zero energy.")


def nmse_db(y: np.ndarray, y_hat: np.ndarray) -> float:
    _check_inputs(y, y_hat)

    sum_y = np.sum(np.abs(y) ** 2)
    sum_error = np.sum(np.abs(y_hat - y) ** 2)

    if not sum_error:
        return -np.inf

    return float(10 * np.log10(sum_error / sum_y))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    _check_inputs(y, y_hat)

    mse = np.mean(np.abs(y_hat - y) ** 2)
    return float(np.sqrt(mse))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    _check_inputs(y, y_hat)

    sum_error = np.sum(np.abs(y_hat - y) ** 2)

    y_mean = np.mean(y)
    sum_total = np.sum(np.abs(y - y_mean) ** 2)

    if not sum_total:
        return 0.0

    return float(1.0 - (sum_error / sum_total))

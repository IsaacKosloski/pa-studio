from typing import Self

import numpy as np
from core.model import Model


class MemoryPolynomial(Model):
    """Memory Polynomial PA model: y[n] = sum_k sum_m theta[k,m] * x[n-m]*|x[n-m]|^(k).
    Columns ordered k-major, m-minor."""

    def __init__(self, K: int, M: int) -> None:
        self.K = K  # ordem de não-linearidade
        self.M = M  # profundidade de memória
        self.theta: np.ndarray | None = None

    def _build_regressors(self, x: np.ndarray) -> np.ndarray:
        """Creates the Phi matrix (N x K*M): each column is x[n-m]*|x[n-m]|^(k).
        Where k = 0..K-1 and m = 0..M-1."""
        N = len(x)
        column_index = 0
        Phi = np.zeros((N, self.K * self.M), dtype=x.dtype)

        for k in range(self.K):
            for m in range(self.M):
                # Creates the delayed signal with zero-padding
                x_delayed = np.zeros_like(x)
                x_delayed[m:] = x[: N - m]  # Shifts the signal by 'm' right positions
                # Creates the regressors' Phi matrix
                Phi[:, column_index] = x_delayed * (np.abs(x_delayed) ** k)
                column_index += 1

        return Phi

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Phi = _build_regressors(x); resolve theta by LS (np.linalg.lstsq)."""
        Phi = self._build_regressors(x)
        self.theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Phi = _build_regressors(x); returns Phi @ theta."""
        if self.theta is None:
            raise AttributeError("MemoryPolynomial must be fit before predict")

        Phi = self._build_regressors(x)
        y_pred: np.ndarray = Phi @ self.theta

        return y_pred

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """PA behavioral model interface: learn x -> y and predict y given x."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Model":
        """Estimate coefficients from input x and measured output y.
        Store them on the instance and return self."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return modeled output y_hat for x using the estimated coefficients."""

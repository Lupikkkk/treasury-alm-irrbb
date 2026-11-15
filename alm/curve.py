"""Nelson-Siegel-Svensson yield curve (the model used by the ECB and many central banks).

y(t) = b0 + b1 * f1(t/tau1) + b2 * (f1(t/tau1) - exp(-t/tau1)) + b3 * (f1(t/tau2) - exp(-t/tau2))
with f1(x) = (1 - exp(-x)) / x.  Rates are continuously compounded, in decimals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def _f1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x > 1e-8
    out[nz] = (1 - np.exp(-x[nz])) / x[nz]
    return out


@dataclass
class NelsonSiegelSvensson:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def zero_rate(self, t) -> np.ndarray:
        t = np.maximum(np.asarray(t, dtype=float), 1e-6)
        x1, x2 = t / self.tau1, t / self.tau2
        return (
            self.beta0
            + self.beta1 * _f1(x1)
            + self.beta2 * (_f1(x1) - np.exp(-x1))
            + self.beta3 * (_f1(x2) - np.exp(-x2))
        )

    def discount_factor(self, t, shift=None) -> np.ndarray:
        """Discount factors, optionally with an additive shift function shift(t) applied to zero rates."""
        t = np.asarray(t, dtype=float)
        z = self.zero_rate(t)
        if shift is not None:
            z = z + shift(t)
        return np.exp(-z * t)

    def forward_rate(self, t) -> np.ndarray:
        """Instantaneous forward rate f(t) = d/dt [t * z(t)] (numerical)."""
        t = np.asarray(t, dtype=float)
        h = 1e-4
        return ((t + h) * self.zero_rate(t + h) - np.maximum(t - h, 0) * self.zero_rate(np.maximum(t - h, 1e-6))) / (
            (t + h) - np.maximum(t - h, 0)
        )

    def params(self) -> dict:
        return dict(beta0=self.beta0, beta1=self.beta1, beta2=self.beta2, beta3=self.beta3,
                    tau1=self.tau1, tau2=self.tau2)


def fit_nss(maturities, yields, n_starts: int = 12, seed: int = 0) -> tuple[NelsonSiegelSvensson, float]:
    """Fit NSS to observed zero yields by non-linear least squares with multiple starts.

    Returns the fitted curve and the RMSE in basis points.
    """
    m = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    rng = np.random.default_rng(seed)

    def resid(p):
        return NelsonSiegelSvensson(*p).zero_rate(m) - y

    lb = [-0.05, -0.15, -0.30, -0.30, 0.05, 0.05]
    ub = [0.20, 0.15, 0.30, 0.30, 15.0, 30.0]
    best = None
    for _ in range(n_starts):
        x0 = [y[-1], y[0] - y[-1], rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
              rng.uniform(0.3, 5.0), rng.uniform(5.0, 20.0)]
        res = least_squares(resid, x0, bounds=(lb, ub))
        if best is None or res.cost < best.cost:
            best = res
    curve = NelsonSiegelSvensson(*best.x)
    rmse_bp = float(np.sqrt(np.mean(best.fun ** 2)) * 1e4)
    return curve, rmse_bp

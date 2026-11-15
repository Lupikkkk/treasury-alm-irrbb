"""Deposit beta: how much of a change in market rates banks pass on to depositors.

The beta is the key behavioural parameter in ALM: it drives NII sensitivity and the value
of the deposit franchise. It is estimated here with a two-step error-correction model
(Engle-Granger), the standard approach in the interest-rate pass-through literature:

    long run :  d_t = a + beta * r_t + u_t
    short run:  Δd_t = c + gamma * u_{t-1} + phi * Δr_t + e_t

beta  = long-run pass-through, gamma < 0 = speed of adjustment, phi = same-month pass-through.

The levels regression is only meaningful if deposit and market rates are cointegrated, so the
residuals u_t are tested for a unit root (Engle-Granger ADF test with MacKinnon (2010) critical
values). Without cointegration the long-run beta may be a spurious regression.

Pass-through is typically asymmetric: banks raise deposit rates slowly when market rates rise
and cut them quickly when market rates fall. The asymmetric error-correction model

    Δd_t = c + gamma_cut * max(u_{t-1}, 0) + gamma_raise * min(u_{t-1}, 0)
             + phi_up * max(Δr_t, 0) + phi_down * min(Δr_t, 0) + e_t

lets the adjustment speed depend on whether the deposit rate sits above (bank cuts) or below
(bank raises) its long-run level, and reports a t-test of gamma_cut = gamma_raise.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

# ECB Data Portal series (monthly, euro area). MIR = MFI interest rate statistics.
ECB_SERIES = {
    "overnight_households": ("MIR", "M.U2.B.L21.A.R.A.2250.EUR.N"),
    "term_households_new": ("MIR", "M.U2.B.L22.F.R.A.2250.EUR.N"),
    "overnight_corporates": ("MIR", "M.U2.B.L21.A.R.A.2240.EUR.N"),
    "euribor_3m": ("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA"),
}


def fetch_ecb_series(name: str, start: str = "2015-01", end: str | None = None) -> pd.Series:
    """Download one of ``ECB_SERIES`` (percent p.a. converted to decimals), monthly from ``start``
    to ``end`` inclusive (YYYY-MM; default: latest). Needs internet."""
    import requests

    flow, key = ECB_SERIES[name]
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata&startPeriod={start}"
    if end:
        url += f"&endPeriod={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(df["OBS_VALUE"].values / 100.0, index=pd.PeriodIndex(df["TIME_PERIOD"], freq="M"), name=name)
    return s.dropna()


# MacKinnon (2010) response-surface coefficients for the Engle-Granger test with two
# variables and a constant: critical value(T) = b0 + b1/T + b2/T^2, for 1%, 5% and 10%.
_EG_CRIT_COEFS = {0.01: (-3.89644, -10.9519, -33.527), 0.05: (-3.33613, -6.1101, -6.823),
                  0.10: (-3.04445, -4.2412, -2.720)}


def _ols(y: np.ndarray, X: np.ndarray, constant: bool = True):
    """OLS returning coefficients, their covariance matrix, residuals and R²."""
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    if constant:
        X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return coef, cov, resid, r2


def _adf_no_constant(u: np.ndarray, max_lags: int, lags: int) -> tuple[float, float]:
    """ADF regression Δu_t = rho*u_{t-1} + sum_i delta_i*Δu_{t-i} + e_t on the last
    len(u) - 1 - max_lags observations. Returns (t-stat of rho, AIC)."""
    du = np.diff(u)
    n = len(du) - max_lags
    y = du[max_lags:]
    X = [u[max_lags:-1]] + [du[max_lags - i:len(du) - i] for i in range(1, lags + 1)]
    coef, cov, resid, _ = _ols(y, np.column_stack(X), constant=False)
    aic = n * np.log(resid @ resid / n) + 2 * (lags + 1)
    return float(coef[0] / np.sqrt(cov[0, 0])), float(aic)


def engle_granger_test(residuals: np.ndarray, max_lags: int | None = None) -> dict:
    """Engle-Granger cointegration test: ADF test (no constant) on the residuals of the
    levels regression, lag length chosen by AIC. Null hypothesis: no cointegration."""
    u = np.asarray(residuals, dtype=float)
    if max_lags is None:
        max_lags = int(np.ceil(12 * (len(u) / 100) ** 0.25))
    max_lags = min(max_lags, len(u) // 2 - 1)
    # choose the lag on a common sample, then re-estimate on all available observations
    lags = min(range(max_lags + 1), key=lambda p: _adf_no_constant(u, max_lags, p)[1])
    t_stat, _ = _adf_no_constant(u, lags, lags)
    T = len(u) - 1  # sample size used for the critical values, as in Stata's egranger and statsmodels
    crit = {lvl: b0 + b1 / T + b2 / T ** 2 for lvl, (b0, b1, b2) in _EG_CRIT_COEFS.items()}
    return {"eg_t_stat": t_stat, "eg_lags": lags, "eg_crit_1pct": crit[0.01], "eg_crit_5pct": crit[0.05],
            "eg_crit_10pct": crit[0.10], "cointegrated_5pct": bool(t_stat < crit[0.05])}


def _half_life(gamma: float) -> float:
    return float(np.log(0.5) / np.log(1 + gamma)) if -1 < gamma < 0 else np.nan


def estimate_deposit_beta(deposit_rate: pd.Series, market_rate: pd.Series) -> dict:
    """Error-correction estimate of deposit pass-through. Inputs: aligned rate levels (decimals)."""
    df = pd.concat([deposit_rate.rename("d"), market_rate.rename("r")], axis=1).dropna()
    d, r = df["d"].values, df["r"].values

    # step 1: long-run (cointegrating) regression and Engle-Granger test on its residuals.
    # Note: OLS standard errors in the levels regression are not valid under cointegration
    # (residuals are autocorrelated); se_beta is reported for reference only.
    (a, beta), cov_long, u, r2_long = _ols(d, r)
    eg = engle_granger_test(u)

    # step 2: symmetric error-correction model
    dd, dr = np.diff(d), np.diff(r)
    (c, gamma, phi), cov_s, _, r2_short = _ols(dd, np.column_stack([u[:-1], dr]))
    se_gamma = np.sqrt(cov_s[1, 1])

    # step 3: asymmetric error-correction model
    ul = u[:-1]
    Xa = np.column_stack([np.maximum(ul, 0), np.minimum(ul, 0), np.maximum(dr, 0), np.minimum(dr, 0)])
    coef_a, cov_a, _, _ = _ols(dd, Xa)
    g_cut, g_raise, phi_up, phi_down = coef_a[1:]
    diff = np.array([0, 1, -1, 0, 0])
    t_asym = float(diff @ coef_a / np.sqrt(diff @ cov_a @ diff))

    # realised pass-through over the hiking phase (trough -> peak) and the cutting phase
    # (peak -> subsequent trough) of the market rate
    i_hi = int(np.argmax(r))
    i_lo = int(np.argmin(r[:i_hi + 1]))
    i_lo2 = i_hi + int(np.argmin(r[i_hi:]))
    up = (d[i_hi] - d[i_lo]) / (r[i_hi] - r[i_lo]) if r[i_hi] > r[i_lo] else np.nan
    down = (d[i_lo2] - d[i_hi]) / (r[i_lo2] - r[i_hi]) if r[i_hi] > r[i_lo2] else np.nan
    return {
        "beta_long_run": float(beta), "se_beta": float(np.sqrt(cov_long[1, 1])), "intercept": float(a),
        **eg,
        "adjustment_speed": float(gamma), "t_stat_adjustment": float(gamma / se_gamma),
        "beta_same_month": float(phi), "half_life_months": _half_life(gamma),
        "adjustment_speed_cut": float(g_cut), "adjustment_speed_raise": float(g_raise),
        "half_life_cut_months": _half_life(g_cut), "half_life_raise_months": _half_life(g_raise),
        "beta_same_month_up": float(phi_up), "beta_same_month_down": float(phi_down),
        "t_stat_asymmetry": t_asym,
        "cycle_beta_trough_to_peak": float(up), "cycle_beta_peak_to_trough": float(down),
        "r2_long_run": float(r2_long), "r2_short_run": float(r2_short), "observations": int(len(df)),
    }


def simulate_deposit_rates(beta: float = 0.3, speed: float = -0.15, n_months: int = 120, noise: float = 0.0002,
                           seed: int = 0, speed_raise: float | None = None) -> tuple[pd.Series, pd.Series]:
    """SIMULATED market and deposit rates (a stylised low-rate period followed by a hiking and
    cutting cycle). For tests and the offline demo only; this is not real data.

    ``speed`` is the adjustment speed when the deposit rate is above its long-run level (the bank
    cuts); ``speed_raise`` the speed when it is below (the bank raises). ``None`` = symmetric."""
    speed_raise = speed if speed_raise is None else speed_raise
    rng = np.random.default_rng(seed)
    r = np.r_[np.full(n_months // 2, -0.004), np.linspace(-0.004, 0.039, n_months // 5),
              np.full(n_months // 10, 0.039), np.linspace(0.039, 0.02, n_months // 10)]
    r = np.r_[r, np.full(n_months - len(r), 0.02)] + rng.normal(0, 0.0003, n_months)
    d = np.empty(n_months)
    d[0] = 0.001 + beta * r[0]
    for t in range(1, n_months):
        gap = d[t - 1] - (0.001 + beta * r[t - 1])
        d[t] = d[t - 1] + (speed if gap > 0 else speed_raise) * gap + 0.05 * (r[t] - r[t - 1]) + rng.normal(0, noise)
    idx = pd.period_range("2016-01", periods=n_months, freq="M")
    return pd.Series(d, index=idx, name="deposit_rate"), pd.Series(r, index=idx, name="market_rate")

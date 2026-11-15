"""Interest Rate Risk in the Banking Book (IRRBB).

Implements the six standardised interest-rate shock scenarios of the Basel Committee
IRRBB standard (BCBS d368, April 2016) and measures:

* ΔEVE: change in the Economic Value of Equity (PV of assets - PV of liabilities)
* ΔNII: change in 12-month Net Interest Income under a constant balance sheet
* key-rate DV01: where on the curve the economic-value risk sits
* hedging: swap portfolios that minimise the worst-case ΔEVE (linear programme)

Shock shapes (t in years, x = 4):
    parallel:  ΔR = ±S_parallel
    short:     ΔR = ±S_short * exp(-t/x)
    long:      ΔR = ±S_long * (1 - exp(-t/x))
    steepener: ΔR = -0.65*|short(t)| + 0.9*|long(t)|
    flattener: ΔR = +0.8*|short(t)| - 0.6*|long(t)|
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .balance_sheet import BalanceSheet, Position
from .curve import NelsonSiegelSvensson

# BCBS standard shock sizes (basis points): parallel, short, long
BCBS_SHOCK_SIZES = {"EUR": (200, 250, 100), "USD": (200, 300, 150), "GBP": (250, 300, 150)}
X = 4.0
SCENARIOS = ["parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"]

# BCBS standardised-framework scalars for conditional prepayment rates: borrowers prepay
# less when rates rise (their loan is cheap) and more when rates fall (they refinance).
PREPAYMENT_SCALARS = {"parallel_up": 0.8, "parallel_down": 1.2, "steepener": 0.8, "flattener": 1.2,
                      "short_up": 0.8, "short_down": 1.2}

# Lowest client rate paid on non-maturity deposits in the NII simulation (zero lower bound
# for retail deposits).
DEPOSIT_RATE_FLOOR = 0.0

Shift = Callable[[np.ndarray], np.ndarray]


def shock_curve(scenario: str, currency: str = "EUR") -> Shift:
    sp, ss, sl = (v / 1e4 for v in BCBS_SHOCK_SIZES[currency])

    def short(t):
        return ss * np.exp(-np.asarray(t, dtype=float) / X)

    def long(t):
        return sl * (1 - np.exp(-np.asarray(t, dtype=float) / X))

    shapes = {
        "base": lambda t: np.zeros_like(np.asarray(t, dtype=float)),
        "parallel_up": lambda t: np.full_like(np.asarray(t, dtype=float), sp),
        "parallel_down": lambda t: np.full_like(np.asarray(t, dtype=float), -sp),
        "short_up": short,
        "short_down": lambda t: -short(t),
        "steepener": lambda t: -0.65 * np.abs(short(t)) + 0.9 * np.abs(long(t)),
        "flattener": lambda t: 0.8 * np.abs(short(t)) - 0.6 * np.abs(long(t)),
    }
    if scenario not in shapes:
        raise ValueError(f"Unknown scenario {scenario}")
    return shapes[scenario]


def floored_shift(curve: NelsonSiegelSvensson, shift: Shift) -> Shift:
    """Apply the post-shock floor: -1.5% at t=0 rising by 3bp per year to 0% at 50y,
    unless the observed rate is already lower (then the observed rate is the floor)."""

    def f(t):
        t = np.asarray(t, dtype=float)
        base = curve.zero_rate(t)
        floor = np.minimum(np.minimum(-0.015 + 0.0003 * t, 0.0), base)
        return np.maximum(base + shift(t), floor) - base

    return f


def _add(shift: Shift | None, extra: Shift) -> Shift:
    return (lambda t: extra(t)) if shift is None else (lambda t: shift(t) + extra(t))


def _pv(p: Position, curve: NelsonSiegelSvensson, shift: Shift | None, cpr_mult: float) -> float:
    cf = p.cash_flows(curve, shift, cpr_mult)
    return float((cf["total"].values * curve.discount_factor(cf["t"].values, shift)).sum())


def present_value(bs: BalanceSheet, curve: NelsonSiegelSvensson, shift: Shift | None = None,
                  cpr_mult: float = 1.0) -> pd.DataFrame:
    """PV and DV01 (PV change for +1bp parallel move) of every position."""
    bump = _add(shift, lambda t: np.full_like(np.asarray(t, dtype=float), 1e-4))
    rows = []
    for p in bs.positions:
        pv = _pv(p, curve, shift, cpr_mult)
        rows.append({"name": p.name, "side": p.side, "notional": p.notional, "pv": pv,
                     "dv01": _pv(p, curve, bump, cpr_mult) - pv, "category": p.category})
    out = pd.DataFrame(rows)
    sign = np.where(out["side"] == "asset", 1.0, -1.0)
    out["signed_pv"] = sign * out["pv"]
    out["signed_dv01"] = sign * out["dv01"]
    return out


def eve(bs: BalanceSheet, curve: NelsonSiegelSvensson, shift: Shift | None = None,
        cpr_mult: float = 1.0) -> float:
    return float(sum(p.sign * _pv(p, curve, shift, cpr_mult) for p in bs.positions))


def eve_sensitivity(bs: BalanceSheet, curve: NelsonSiegelSvensson, currency: str = "EUR",
                    apply_floor: bool = True, behavioural_prepayment: bool = True) -> pd.DataFrame:
    """ΔEVE under the six BCBS scenarios. With ``behavioural_prepayment`` the prepayment speed
    responds to the scenario, which makes fixed-rate loans negatively convex."""
    base = eve(bs, curve)
    rows = []
    for sc in SCENARIOS:
        s = shock_curve(sc, currency)
        if apply_floor:
            s = floored_shift(curve, s)
        val = eve(bs, curve, s, PREPAYMENT_SCALARS[sc] if behavioural_prepayment else 1.0)
        rows.append({"scenario": sc, "EVE": val, "ΔEVE": val - base})
    return pd.DataFrame(rows).set_index("scenario")


def key_rate_dv01(bs: BalanceSheet, curve: NelsonSiegelSvensson,
                  tenors=(0.25, 1, 2, 3, 5, 7, 10, 15, 20, 30)) -> pd.Series:
    """Net DV01 per key rate (EUR per +1bp at that tenor only). Bumps are triangular
    ("hat") functions that sum to a parallel 1bp move, so key-rate DV01s add up to total DV01."""
    tenors = np.asarray(tenors, dtype=float)
    base = eve(bs, curve)
    out = {}
    for i, k in enumerate(tenors):
        xp = tenors
        fp = np.zeros(len(tenors))
        fp[i] = 1e-4
        out[k] = eve(bs, curve, lambda t, fp=fp: np.interp(np.asarray(t, dtype=float), xp, fp)) - base
    return pd.Series(out, name="key_rate_dv01")


# --------------------------------------------------------------------------- NII

def nii(bs: BalanceSheet, curve: NelsonSiegelSvensson, shift: Shift | None = None, horizon: float = 1.0,
        cpr_mult: float = 1.0, deposit_rate_floor: float | None = DEPOSIT_RATE_FLOOR) -> float:
    """12-month NII under a constant balance sheet.

    * Fixed-rate positions earn their coupon; principal repaid (scheduled or prepaid) within the
      horizon is reinvested in the same product at the original rate plus the shift at that tenor.
    * Floating positions earn the current rate until the next reset, then the shocked rate.
    * Non-maturity deposits reprice immediately by ``beta`` x the short-rate shift (deposit beta),
      but not below ``deposit_rate_floor`` (banks do not charge retail depositors negative rates).
      If the client rate is already below the floor, the current rate is the floor. ``None`` = no floor.
    """
    def sh(t: float) -> float:
        return float(np.asarray(shift(np.array([t])))[0]) if shift else 0.0

    total = 0.0
    for p in bs.positions:
        if p.is_nmd:
            client_rate = p.rate + (p.beta or 0.0) * sh(0.25)
            if deposit_rate_floor is not None and p.side == "liability":
                client_rate = max(client_rate, min(deposit_rate_floor, p.rate))
            income = p.notional * client_rate * horizon
        elif p.fixed:
            cf = p.cash_flows(curve, None, cpr_mult)
            within = cf[cf["t"] <= horizon + 1e-9]
            new_rate = p.rate + sh(p.maturity)
            income = within["interest"].sum() + (within["principal"] * new_rate * (horizon - within["t"])).sum()
        else:
            reset = min(p.reset_freq, p.maturity, horizon)
            base = float(curve.zero_rate(np.array([max(p.reset_freq, 0.01)]))[0])
            income = p.notional * ((base + p.rate) * reset + (base + sh(p.reset_freq) + p.rate) * (horizon - reset))
        total += p.sign * income
    return total


def nii_sensitivity(bs: BalanceSheet, curve: NelsonSiegelSvensson, currency: str = "EUR") -> pd.DataFrame:
    base = nii(bs, curve)
    rows = []
    for sc in ["parallel_up", "parallel_down", "short_up", "short_down"]:
        val = nii(bs, curve, shock_curve(sc, currency), cpr_mult=PREPAYMENT_SCALARS[sc])
        rows.append({"scenario": sc, "NII": val, "ΔNII": val - base})
    return pd.DataFrame(rows).set_index("scenario")


# ----------------------------------------------------------------------- hedging

def _swap_legs(notional: float, tenor: float, fixed_rate: float) -> list[Position]:
    """Pay-fixed swap = long 3M floater + short fixed bond. Negative notional = receive-fixed."""
    a, l = ("asset", "liability") if notional >= 0 else ("liability", "asset")
    kind = "payer" if notional >= 0 else "receiver"
    n = abs(notional)
    return [
        Position(f"{tenor:g}y {kind} swap: floating leg", a, n, 0.0, tenor, fixed=False, reset_freq=0.25,
                 category="Hedge"),
        Position(f"{tenor:g}y {kind} swap: fixed leg", l, n, fixed_rate, tenor, pay_freq=1.0, category="Hedge"),
    ]


def par_swap_rate(curve: NelsonSiegelSvensson, tenor: float) -> float:
    """Annual-pay par swap rate from the zero curve: (1 - DF_T) / sum(DF_i)."""
    t = np.arange(1, int(round(tenor)) + 1, dtype=float)
    df = curve.discount_factor(t)
    return float((1 - df[-1]) / df.sum())


def add_swap(bs: BalanceSheet, notional: float, tenor: float, fixed_rate: float | None = None,
             curve: NelsonSiegelSvensson | None = None) -> BalanceSheet:
    """Copy of ``bs`` with an interest-rate swap added (positive notional = pay fixed)."""
    if fixed_rate is None:
        fixed_rate = par_swap_rate(curve, tenor)
    return BalanceSheet(list(bs.positions) + (_swap_legs(notional, tenor, fixed_rate) if notional else []))


add_payer_swap = add_swap  # backwards-compatible name


def _unit_swap_deltas(curve: NelsonSiegelSvensson, tenors, currency: str) -> pd.DataFrame:
    """ΔEVE per scenario of a pay-fixed swap with notional 1, for each tenor."""
    return pd.DataFrame({
        T: eve_sensitivity(BalanceSheet(_swap_legs(1.0, T, par_swap_rate(curve, T))), curve, currency)["ΔEVE"]
        for T in tenors
    })


def size_hedge(bs: BalanceSheet, curve: NelsonSiegelSvensson, tier1: float, tenor: float = 10.0,
               target: float = 0.15, currency: str = "EUR") -> dict:
    """Smallest single pay-fixed swap that keeps the worst ΔEVE within ``target`` x Tier 1.

    ΔEVE is linear in the swap notional, so each scenario gives a linear inequality and the
    feasible notionals form an interval. If it is empty, the notional that maximises the worst
    ΔEVE is returned with ``target_met=False``.
    """
    d0 = eve_sensitivity(bs, curve, currency)["ΔEVE"]
    d1 = _unit_swap_deltas(curve, [tenor], currency)[tenor]
    limit = -target * tier1
    lo, hi = 0.0, np.inf
    for s in SCENARIOS:
        if d1[s] > 0:
            lo = max(lo, (limit - d0[s]) / d1[s])
        elif d1[s] < 0:
            hi = min(hi, (limit - d0[s]) / d1[s])
    met = lo <= hi
    if met:
        n = lo
    else:  # maximise the worst case: max t s.t. d0_s + n*d1_s >= t
        res = linprog(c=[0, -1], A_ub=np.c_[-d1.values, np.ones(len(d1))], b_ub=d0.values,
                      bounds=[(0, None), (None, None)])
        n = float(res.x[0])
    worst = float((d0 + n * d1).min())
    return {"notional": float(n), "tenor": tenor, "rate": par_swap_rate(curve, tenor), "worst_ΔEVE": worst,
            "target_met": bool(met)}


def optimise_hedge(bs: BalanceSheet, curve: NelsonSiegelSvensson, tier1: float, limit: float = 0.10,
                   tenors=(2, 5, 7, 10, 15, 20), currency: str = "EUR") -> dict:
    """Cheapest swap portfolio (smallest gross notional, payers and receivers across ``tenors``)
    that keeps ΔEVE within ``limit`` x Tier 1 in every one of the six scenarios.

    Linear programme (ΔEVE is linear in swap notionals):
        min  sum_i (p_i + q_i)
        s.t. d0_s + sum_i (p_i - q_i) * d_is >= -limit * Tier1   for every scenario s
             p_i, q_i >= 0        (p = pay-fixed notional, q = receive-fixed notional)
    """
    d0 = eve_sensitivity(bs, curve, currency)["ΔEVE"]
    D = _unit_swap_deltas(curve, tenors, currency)
    k = len(tenors)
    res = linprog(np.ones(2 * k), A_ub=np.c_[-D.values, D.values], b_ub=d0.values + limit * tier1,
                  bounds=[(0, None)] * (2 * k))
    if not res.success:
        return {"feasible": False, "notionals": pd.Series(0.0, index=list(tenors), name="notional"),
                "ΔEVE_after": d0, "worst_ΔEVE": float(d0.min()), "gross_notional": 0.0}
    net = res.x[:k] - res.x[k:]
    after = d0 + D.values @ net
    return {"feasible": True, "notionals": pd.Series(net, index=list(tenors), name="notional").round(1),
            "ΔEVE_after": after, "worst_ΔEVE": float(after.min()), "gross_notional": float(np.abs(net).sum())}


def apply_hedge(bs: BalanceSheet, curve: NelsonSiegelSvensson, notionals: pd.Series) -> BalanceSheet:
    out = bs
    for tenor, n in notionals.items():
        if abs(n) > 1e-9:
            out = add_swap(out, float(n), float(tenor), curve=curve)
    return out


# ------------------------------------------------------------------------ report

def irrbb_report(bs: BalanceSheet, curve: NelsonSiegelSvensson, tier1: float, currency: str = "EUR") -> dict:
    """Headline IRRBB metrics, including the Supervisory Outlier Tests used in the EU:
    ΔEVE worse than -15% of Tier 1 (six scenarios), ΔNII worse than -5% of Tier 1 (parallel ±)."""
    eve_tab = eve_sensitivity(bs, curve, currency)
    nii_tab = nii_sensitivity(bs, curve, currency)
    worst_eve = float(eve_tab["ΔEVE"].min())
    worst_nii = float(nii_tab.loc[["parallel_up", "parallel_down"], "ΔNII"].min())
    return {
        "base_EVE": eve(bs, curve),
        "base_NII": nii(bs, curve),
        "eve_table": eve_tab,
        "nii_table": nii_tab,
        "worst_ΔEVE": worst_eve,
        "worst_scenario": eve_tab["ΔEVE"].idxmin(),
        "ΔEVE_pct_tier1": worst_eve / tier1,
        "SOT_outlier_15pct": worst_eve / tier1 < -0.15,
        "worst_ΔNII": worst_nii,
        "ΔNII_pct_tier1": worst_nii / tier1,
        "SOT_NII_outlier_5pct": worst_nii / tier1 < -0.05,
    }

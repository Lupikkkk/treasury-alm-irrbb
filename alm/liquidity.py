"""Liquidity metrics: a simplified Liquidity Coverage Ratio (Basel III / CRR) and a
deposit-run survival-horizon stress test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .balance_sheet import BalanceSheet

HQLA_HAIRCUT = {"L1": 0.0, "L2A": 0.15, "L2B": 0.50}


def lcr(bs: BalanceSheet, outflow_multiplier: float = 1.0) -> dict:
    """LCR = HQLA / (outflows - min(inflows, 75% of outflows)) over a 30-day stress.

    Applies standard haircuts and the 40% cap on Level 2 assets (simplified: cap applied
    to post-haircut Level 2 relative to total HQLA).
    """
    l1 = sum(p.notional for p in bs.positions if p.hqla_level == "L1")
    l2a = sum(p.notional * (1 - HQLA_HAIRCUT["L2A"]) for p in bs.positions if p.hqla_level == "L2A")
    l2b = sum(p.notional * (1 - HQLA_HAIRCUT["L2B"]) for p in bs.positions if p.hqla_level == "L2B")
    l2 = min(l2a + l2b, 2 / 3 * l1)  # Level 2 <= 40% of total HQLA  <=>  L2 <= 2/3 * L1
    hqla = l1 + l2

    outflows = sum(p.notional * min(p.runoff_30d * outflow_multiplier, 1.0)
                   for p in bs.positions if p.side == "liability")
    inflows = sum(p.notional * p.runoff_30d for p in bs.positions if p.side == "asset" and not p.hqla_level)
    net = outflows - min(inflows, 0.75 * outflows)
    ratio = hqla / net if net > 0 else np.inf
    return {"HQLA": hqla, "level1": l1, "level2_after_cap": l2, "outflows_30d": outflows,
            "inflows_30d": inflows, "net_outflows": net, "LCR": ratio, "meets_100pct": ratio >= 1.0}


def survival_horizon(bs: BalanceSheet, daily_runoff_multiplier: float = 1.0, max_days: int = 365) -> pd.DataFrame:
    """Day-by-day stress: liabilities run off at a constant daily rate implied by their 30-day
    run-off assumption (scaled by ``daily_runoff_multiplier``); the bank pays out from cash first,
    then from HQLA (after haircut). Returns the liquidity buffer path; survival = first day < 0."""
    buffer = sum(p.notional * (1 - HQLA_HAIRCUT.get(p.hqla_level, 1.0))
                 for p in bs.positions if p.hqla_level)
    liabs = [(p.notional, 1 - (1 - min(p.runoff_30d * daily_runoff_multiplier, 1.0)) ** (1 / 30))
             for p in bs.positions if p.side == "liability"]
    bal = np.array([n for n, _ in liabs], dtype=float)
    rates = np.array([r for _, r in liabs])
    path = []
    for day in range(1, max_days + 1):
        out = bal * rates
        bal -= out
        buffer -= out.sum()
        path.append({"day": day, "outflow": out.sum(), "buffer": buffer})
        if buffer < 0:
            break
    return pd.DataFrame(path).set_index("day")

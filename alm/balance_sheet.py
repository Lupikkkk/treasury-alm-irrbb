"""Balance-sheet positions and cash-flow generation.

Each position is a homogeneous bucket (e.g. "fixed-rate mortgages") described by notional,
coupon, maturity, repricing frequency and amortisation profile. Non-maturity deposits (NMDs)
are modelled with a behavioural maturity, as banks do in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Position:
    name: str
    side: str  # "asset" or "liability"
    notional: float
    rate: float  # annual coupon / client rate (for floaters: spread over the reference rate)
    maturity: float  # years (behavioural for NMDs)
    fixed: bool = True  # fixed-rate or floating
    reset_freq: float = 0.25  # years between rate resets for floaters
    amortising: bool = False  # linear amortisation if True, bullet otherwise
    pay_freq: float = 0.25  # years between payments
    hqla_level: str | None = None  # "L1", "L2A", "L2B" for LCR purposes
    runoff_30d: float = 0.0  # LCR outflow (liabilities) / inflow (assets) rate
    category: str = ""
    cpr: float = 0.0  # constant prepayment rate p.a. (fixed-rate loans)
    core_share: float | None = None  # NMDs: share of balances treated as stable ("core")
    beta: float | None = None  # NMDs: pass-through of market-rate changes to the client rate

    @property
    def sign(self) -> int:
        return 1 if self.side == "asset" else -1

    @property
    def is_nmd(self) -> bool:
        return self.core_share is not None

    def cash_flows(self, curve=None, shift=None, cpr_mult: float = 1.0) -> pd.DataFrame:
        """Contractual or behavioural cash flows: time, interest and principal.

        * Fixed-rate: bullet or linear amortisation; with ``cpr`` > 0 the outstanding balance
          also prepays at a constant annual rate (scaled by ``cpr_mult`` in shock scenarios).
        * Floating: repriced at the next reset date, where the position is worth par, so only
          cash flows up to the first reset are generated (standard repricing-gap treatment).
        * Non-maturity deposits: the non-core share reprices overnight; the core share runs
          off linearly over the behavioural ``maturity`` (average life = maturity / 2).
        """
        if self.is_nmd:
            core = self.notional * self.core_share
            n = max(int(round(self.maturity / self.pay_freq)), 1)
            dt = self.maturity / n
            times = np.arange(1, n + 1) * dt
            bal_start = core * (1 - (times - dt) / self.maturity)
            cf = pd.DataFrame({"t": times, "interest": bal_start * self.rate * dt,
                               "principal": np.full(n, core / n)})
            overnight = pd.DataFrame({"t": [1 / 365], "interest": [0.0],
                                      "principal": [self.notional - core]})
            cf = pd.concat([overnight, cf], ignore_index=True)
            cf["total"] = cf["interest"] + cf["principal"]
            return cf

        horizon = self.maturity if self.fixed else min(self.reset_freq, self.maturity)
        n = max(int(round(horizon / self.pay_freq)), 1)
        dt = horizon / n
        times = np.arange(1, n + 1) * dt

        if self.fixed:
            rate = self.rate
        else:
            base = 0.0 if curve is None else float(curve.zero_rate(np.array([self.reset_freq]))[0])
            if shift is not None:
                base += float(np.asarray(shift(np.array([self.reset_freq])))[0])
            rate = base + self.rate

        k = np.arange(0, n + 1)
        scheduled = 1 - k / n if (self.amortising and self.fixed) else np.r_[np.ones(n), 0.0]
        survival = (1 - min(self.cpr * cpr_mult, 0.99)) ** (k * dt) if self.fixed else np.ones(n + 1)
        balance = self.notional * scheduled * survival
        principal = balance[:-1] - balance[1:]
        interest = balance[:-1] * rate * dt
        return pd.DataFrame({"t": times, "interest": interest, "principal": principal,
                             "total": interest + principal})

    def average_life(self) -> float:
        cf = self.cash_flows()
        return float((cf["t"] * cf["principal"]).sum() / cf["principal"].sum())


@dataclass
class BalanceSheet:
    positions: list[Position] = field(default_factory=list)

    def add(self, p: Position) -> None:
        self.positions.append(p)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "name": p.name, "side": p.side, "notional": p.notional, "rate": p.rate,
            "maturity": p.maturity, "fixed": p.fixed, "avg_life": round(p.average_life(), 2),
            "cpr": p.cpr, "core_share": p.core_share, "beta": p.beta, "category": p.category,
        } for p in self.positions])

    def totals(self) -> dict:
        a = sum(p.notional for p in self.positions if p.side == "asset")
        l = sum(p.notional for p in self.positions if p.side == "liability")
        return {"assets": a, "liabilities": l, "equity": a - l}

    def repricing_gap(self, buckets=(0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)) -> pd.DataFrame:
        """Notional repricing in each time bucket (assets minus liabilities)."""
        rows = []
        for p in self.positions:
            if p.fixed or p.is_nmd:
                cf = p.cash_flows()
                rows += [(t, p.sign * pr) for t, pr in zip(cf["t"], cf["principal"])]
            else:
                rows.append((min(p.reset_freq, p.maturity), p.sign * p.notional))
        df = pd.DataFrame(rows, columns=["t", "amount"])
        labels = [f"{buckets[i]}-{buckets[i + 1]}y" for i in range(len(buckets) - 1)]
        df["bucket"] = pd.cut(df["t"], bins=list(buckets), labels=labels, include_lowest=True)
        gap = df.groupby("bucket", observed=False)["amount"].sum().to_frame("gap")
        gap["cumulative_gap"] = gap["gap"].cumsum()
        return gap


def sample_bank() -> BalanceSheet:
    """An illustrative mid-sized European retail bank (EUR millions). Figures are made up,
    chosen to look like a typical balance-sheet mix, not taken from any real institution."""
    bs = BalanceSheet()
    A, L = "asset", "liability"
    # --- Assets ---
    bs.add(Position("Cash & central bank reserves", A, 600, 0.0300, 0.01, fixed=False, reset_freq=0.01,
                    hqla_level="L1", category="Liquidity"))
    bs.add(Position("Government bonds (5y)", A, 900, 0.0275, 5, pay_freq=1.0, hqla_level="L1",
                    category="Securities"))
    bs.add(Position("Covered bonds (3y)", A, 300, 0.0310, 3, pay_freq=1.0, hqla_level="L2A",
                    category="Securities"))
    bs.add(Position("Fixed-rate mortgages (20y)", A, 2600, 0.0380, 20, amortising=True, pay_freq=1 / 12,
                    runoff_30d=0.005, cpr=0.04, category="Loans"))
    bs.add(Position("Floating mortgages (Euribor 6M + 1.6%)", A, 1200, 0.0160, 25, fixed=False, reset_freq=0.5,
                    category="Loans"))
    bs.add(Position("SME loans (Euribor 3M + 2.5%)", A, 1100, 0.0250, 4, fixed=False, reset_freq=0.25,
                    runoff_30d=0.05, category="Loans"))
    bs.add(Position("Consumer loans (fixed, 4y)", A, 400, 0.0790, 4, amortising=True, pay_freq=1 / 12,
                    runoff_30d=0.02, cpr=0.08, category="Loans"))
    # --- Liabilities ---
    # NMDs: core share and behavioural maturity respect the BCBS caps for retail transactional
    # deposits (core <= 90%, average life of core <= 5y) and non-transactional (70%, 4.5y).
    bs.add(Position("Retail sight deposits (transactional, NMD)", L, 2500, 0.0025, 9.0, pay_freq=0.25,
                    runoff_30d=0.05, core_share=0.85, beta=0.15, category="Deposits"))
    bs.add(Position("Retail savings deposits (non-transactional, NMD)", L, 900, 0.0100, 6.0, pay_freq=0.25,
                    runoff_30d=0.10, core_share=0.65, beta=0.45, category="Deposits"))
    bs.add(Position("Term deposits (1y)", L, 1000, 0.0260, 1.0, pay_freq=1.0, runoff_30d=0.10,
                    category="Deposits"))
    bs.add(Position("Corporate operational deposits", L, 600, 0.0100, 0.25, fixed=False, reset_freq=0.08,
                    runoff_30d=0.25, category="Deposits"))
    bs.add(Position("Corporate non-operational deposits", L, 400, 0.0200, 0.08, fixed=False, reset_freq=0.08,
                    runoff_30d=0.40, category="Deposits"))
    bs.add(Position("Senior unsecured bond (5y)", L, 500, 0.0350, 5, pay_freq=1.0, category="Wholesale"))
    bs.add(Position("Covered bond issued (7y)", L, 600, 0.0300, 7, pay_freq=1.0, category="Wholesale"))
    return bs

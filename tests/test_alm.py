import numpy as np
import pytest

from alm import NelsonSiegelSvensson, Position, BalanceSheet, fit_nss, lcr, sample_bank, shock_curve
from alm.data import sample_curve
from alm.irrbb import eve, eve_sensitivity, nii, present_value


@pytest.fixture(scope="module")
def curve():
    c = sample_curve()
    fitted, _ = fit_nss(c.index.values, c.values)
    return fitted


def test_nss_fit_is_accurate():
    c = sample_curve()
    _, rmse = fit_nss(c.index.values, c.values)
    assert rmse < 5  # bp


def test_nss_recovers_known_parameters():
    true = NelsonSiegelSvensson(0.03, -0.01, 0.01, -0.005, 1.5, 8.0)
    m = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    fitted, rmse = fit_nss(m, true.zero_rate(m))
    assert rmse < 0.5
    assert np.allclose(fitted.zero_rate(m), true.zero_rate(m), atol=1e-4)


def test_zero_coupon_bond_pv_matches_discount_factor():
    flat = NelsonSiegelSvensson(0.03, 0, 0, 0, 1, 10)
    p = Position("ZCB", "asset", 100, 0.0, 5, pay_freq=5)
    bs = BalanceSheet([p])
    assert eve(bs, flat) == pytest.approx(100 * np.exp(-0.03 * 5), rel=1e-9)


def test_par_floater_insensitive_to_rates():
    flat = NelsonSiegelSvensson(0.03, 0, 0, 0, 1, 10)
    fl = Position("Floater", "asset", 100, 0.0, 10, fixed=False, reset_freq=0.25)
    bs = BalanceSheet([fl])
    dv01 = present_value(bs, flat)["dv01"].iloc[0]
    fixed = BalanceSheet([Position("Bond", "asset", 100, 0.03, 10, pay_freq=1)])
    dv01_fixed = present_value(fixed, flat)["dv01"].iloc[0]
    assert abs(dv01) < abs(dv01_fixed) / 20


def test_long_fixed_assets_lose_value_when_rates_rise(curve):
    bs = BalanceSheet([Position("Mortgage", "asset", 100, 0.04, 20, amortising=True, pay_freq=1 / 12)])
    tab = eve_sensitivity(bs, curve, apply_floor=False)
    assert tab.loc["parallel_up", "ΔEVE"] < 0 < tab.loc["parallel_down", "ΔEVE"]


def test_bcbs_shock_shapes():
    t = np.array([0.0, 4.0, 100.0])
    assert shock_curve("parallel_up")(t) == pytest.approx([0.02] * 3)
    assert shock_curve("short_up")(t)[0] == pytest.approx(0.025)
    assert shock_curve("short_up")(t)[2] == pytest.approx(0.0, abs=1e-9)
    assert shock_curve("steepener")(t)[0] == pytest.approx(-0.65 * 0.025)
    assert shock_curve("steepener")(t)[2] == pytest.approx(0.9 * 0.01, rel=1e-6)


def test_floating_assets_raise_nii_when_rates_rise(curve):
    bs = BalanceSheet([Position("SME", "asset", 100, 0.02, 3, fixed=False, reset_freq=0.25)])
    assert nii(bs, curve, shock_curve("parallel_up")) > nii(bs, curve)


def test_lcr_level2_cap():
    bs = BalanceSheet([
        Position("L1", "asset", 10, 0.0, 1, hqla_level="L1"),
        Position("L2A", "asset", 1000, 0.0, 1, hqla_level="L2A"),
        Position("Dep", "liability", 100, 0.0, 1, runoff_30d=0.5),
    ])
    out = lcr(bs)
    assert out["level2_after_cap"] == pytest.approx(10 * 2 / 3)


def test_sample_bank_runs(curve):
    bs = sample_bank()
    assert bs.totals()["equity"] > 0
    assert np.isfinite(eve(bs, curve))
    assert lcr(bs)["LCR"] > 0


# ---- behavioural models, key rates, hedging, deposit betas ----

def test_prepayment_shortens_average_life_and_duration(curve):
    slow = Position("M", "asset", 100, 0.04, 20, amortising=True, pay_freq=1 / 12)
    fast = Position("M", "asset", 100, 0.04, 20, amortising=True, pay_freq=1 / 12, cpr=0.06)
    assert fast.average_life() < slow.average_life()
    assert fast.cash_flows()["principal"].sum() == pytest.approx(100)
    dv = lambda p: abs(present_value(BalanceSheet([p]), curve)["dv01"].iloc[0])
    assert dv(fast) < dv(slow)


def test_behavioural_prepayment_creates_negative_convexity(curve):
    bs = BalanceSheet([Position("M", "asset", 100, 0.04, 20, amortising=True, pay_freq=1 / 12, cpr=0.05)])
    static = eve_sensitivity(bs, curve, apply_floor=False, behavioural_prepayment=False)["ΔEVE"]
    behav = eve_sensitivity(bs, curve, apply_floor=False, behavioural_prepayment=True)["ΔEVE"]
    assert behav["parallel_up"] < static["parallel_up"]      # lose more when rates rise
    assert behav["parallel_down"] < static["parallel_down"]  # gain less when rates fall


def test_nmd_cash_flows_and_core_duration(curve):
    nmd = Position("NMD", "liability", 100, 0.0, 8, pay_freq=0.25, core_share=0.8, beta=0.2)
    cf = nmd.cash_flows()
    assert cf["principal"].sum() == pytest.approx(100)
    assert cf["principal"].iloc[0] == pytest.approx(20)  # non-core reprices overnight
    more_core = Position("NMD", "liability", 100, 0.0, 8, pay_freq=0.25, core_share=0.9, beta=0.2)
    dv = lambda p: abs(present_value(BalanceSheet([p]), curve)["dv01"].iloc[0])
    assert dv(more_core) > dv(nmd)


def test_deposit_beta_reduces_nii_benefit_of_higher_rates(curve):
    def bank(beta):
        return BalanceSheet([
            Position("Loans", "asset", 100, 0.02, 3, fixed=False, reset_freq=0.25),
            Position("NMD", "liability", 100, 0.0, 6, pay_freq=0.25, core_share=0.8, beta=beta),
        ])
    up = shock_curve("parallel_up")
    gain = lambda b: nii(bank(b), curve, up) - nii(bank(b), curve)
    assert gain(0.0) > gain(0.5) > gain(1.0)
    assert gain(1.0) == pytest.approx(-100 * 0.02 * 0.25, abs=0.01)  # only the reset lag remains


def test_key_rate_dv01_sums_to_parallel_dv01(curve):
    from alm import key_rate_dv01
    bs = sample_bank()
    assert key_rate_dv01(bs, curve).sum() == pytest.approx(present_value(bs, curve)["signed_dv01"].sum(), rel=1e-3)


def test_par_swap_has_zero_value_and_negative_dv01_for_receiver(curve):
    from alm import add_swap
    payer = add_swap(BalanceSheet([]), 100, 10, curve=curve)
    assert eve(payer, curve) == pytest.approx(0.0, abs=0.05)
    assert present_value(payer, curve)["signed_dv01"].sum() > 0  # pay-fixed gains when rates rise


def test_hedge_optimiser_meets_limit_and_is_linear(curve):
    from alm import optimise_hedge, size_hedge
    from alm.irrbb import apply_hedge
    bs, tier1 = sample_bank(), 550.0
    opt = optimise_hedge(bs, curve, tier1, limit=0.10)
    assert opt["feasible"] and opt["worst_ΔEVE"] >= -0.10 * tier1 - 1e-6
    full = eve_sensitivity(apply_hedge(bs, curve, opt["notionals"]), curve)["ΔEVE"]
    assert np.allclose(full.values, opt["ΔEVE_after"].values, atol=0.1)  # LP prediction = full revaluation
    single = size_hedge(bs, curve, tier1, tenor=10.0, target=0.10)
    assert opt["gross_notional"] <= single["notional"] + 1e-6


def test_deposit_beta_estimator_recovers_true_beta():
    from alm import estimate_deposit_beta, simulate_deposit_rates
    d, r = simulate_deposit_rates(beta=0.35, speed=-0.2, n_months=240, seed=3)
    est = estimate_deposit_beta(d, r)
    assert est["beta_long_run"] == pytest.approx(0.35, abs=0.04)
    assert est["adjustment_speed"] < 0 and est["t_stat_adjustment"] < -3


# ---- deposit-rate floor, cointegration and asymmetric pass-through ----

def test_deposit_rate_floored_at_zero_in_down_shock(curve):
    # client rate 0.25%, beta 0.5, -200bp: unfloored rate would be -0.75%
    dep = Position("NMD", "liability", 100, 0.0025, 6, pay_freq=0.25, core_share=0.8, beta=0.5)
    bs = BalanceSheet([dep])
    down = shock_curve("parallel_down")
    assert nii(bs, curve, down) == pytest.approx(-100 * 0.0)  # cost cannot fall below zero
    assert nii(bs, curve, down, deposit_rate_floor=None) == pytest.approx(-100 * (0.0025 - 0.5 * 0.02))
    # a client rate already below the floor stays where it is
    neg = BalanceSheet([Position("NMD", "liability", 100, -0.002, 6, pay_freq=0.25, core_share=0.8, beta=0.5)])
    assert nii(neg, curve, down) == pytest.approx(-100 * -0.002)


def _random_walk_deposits(n=300, beta=0.4, g_cut=-0.5, g_raise=-0.1, seed=0):
    """Market rate = random walk (I(1)); deposit rate error-corrects to 0.1% + beta * r with
    a speed that depends on the sign of the gap (threshold ECM)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    r = 0.02 + np.cumsum(rng.normal(0, 0.002, n))
    d = np.empty(n)
    d[0] = 0.001 + beta * r[0]
    for t in range(1, n):
        gap = d[t - 1] - (0.001 + beta * r[t - 1])
        d[t] = d[t - 1] + (g_cut if gap > 0 else g_raise) * gap + rng.normal(0, 0.0003)
    return pd.Series(d), pd.Series(r)


def test_engle_granger_detects_cointegration_and_rejects_spurious_regression():
    from alm import estimate_deposit_beta
    est = estimate_deposit_beta(*_random_walk_deposits())
    assert est["cointegrated_5pct"] and est["eg_t_stat"] < est["eg_crit_1pct"]
    import pandas as pd
    rng = np.random.default_rng(1)
    x, y = (pd.Series(np.cumsum(rng.normal(0, 0.001, 240))) for _ in range(2))
    assert not estimate_deposit_beta(y, x)["cointegrated_5pct"]  # two independent random walks


def test_engle_granger_matches_statsmodels():
    coint = pytest.importorskip("statsmodels.tsa.stattools").coint
    from alm import estimate_deposit_beta, simulate_deposit_rates
    for d, r in [_random_walk_deposits(seed=2), simulate_deposit_rates(beta=0.45, seed=2)]:
        est = estimate_deposit_beta(d, r)
        t, _, crit = coint(d.values, r.values)
        assert est["eg_t_stat"] == pytest.approx(t, rel=1e-8)
        assert est["eg_crit_5pct"] == pytest.approx(crit[1], rel=1e-8)


def test_asymmetric_ecm_detects_slow_raising_and_fast_cutting():
    from alm import estimate_deposit_beta
    est = estimate_deposit_beta(*_random_walk_deposits(g_cut=-0.5, g_raise=-0.1))
    assert est["adjustment_speed_cut"] < est["adjustment_speed_raise"] < 0
    assert est["half_life_raise_months"] > est["half_life_cut_months"]
    assert est["t_stat_asymmetry"] < -2
    sym = estimate_deposit_beta(*_random_walk_deposits(g_cut=-0.25, g_raise=-0.25))
    assert abs(sym["t_stat_asymmetry"]) < 2

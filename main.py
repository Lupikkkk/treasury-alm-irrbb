"""Treasury ALM report: yield curve, IRRBB (EVE / NII), hedging, deposit betas and liquidity.

    python main.py            # illustrative curve and simulated deposit data, runs offline
    python main.py --ecb      # ECB AAA euro-area curve and ECB deposit-rate statistics (internet)

The offline run writes charts, tables and reports/summary.md (every number the README quotes) to
reports/. The --ecb run writes the same files to reports/ecb/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alm import fit_nss, irrbb_report, lcr, sample_bank, shock_curve, survival_horizon
from alm.data import fetch_ecb_curve, sample_curve
from alm.deposits import estimate_deposit_beta, fetch_ecb_series, simulate_deposit_rates
from alm.irrbb import (SCENARIOS, add_swap, apply_hedge, eve_sensitivity, floored_shift, key_rate_dv01,
                       nii_sensitivity, optimise_hedge, present_value, size_hedge)

REPORTS = Path(__file__).parent / "reports"
TIER1 = 550.0  # EUR m, illustrative Tier 1 capital of the sample bank


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _fmt(df: pd.DataFrame, formats: dict[str, str], default: str = "{:,.1f}") -> str:
    """Plain-text table with a number format per column (used for reports/summary.md)."""
    out = df.copy().astype(object)
    for col in df.columns:
        f = formats.get(col, default)
        out[col] = [f.format(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
                    else v for v in df[col]]
    return out.to_string()


def main() -> None:
    p = argparse.ArgumentParser(description="Treasury ALM / IRRBB report")
    p.add_argument("--ecb", action="store_true", help="Use ECB data (yield curve and deposit rates)")
    p.add_argument("--date", default=None, help="Curve date for --ecb (YYYY-MM-DD; default: latest)")
    p.add_argument("--end", default=None, help="Last month of ECB deposit-rate data, inclusive (YYYY-MM)")
    args = p.parse_args()
    out_dir = REPORTS / "ecb" if args.ecb else REPORTS

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ curve
    market = fetch_ecb_curve(args.date) if args.ecb else sample_curve()
    source = "ECB AAA euro-area spot curve" if args.ecb else "illustrative sample curve (not market data)"
    curve, rmse = fit_nss(market.index.values, market.values)
    section(f"Yield curve: {source}")
    print("NSS parameters:", {k: round(float(v), 4) for k, v in curve.params().items()})
    print(f"Fit RMSE: {rmse:.2f} bp")

    # ---------------------------------------------------------- deposit betas
    bs = sample_bank()
    section("Deposit beta estimation (error-correction model)")
    by_sample = None
    if args.ecb:
        long = {k: fetch_ecb_series(k, start="2003-01", end=args.end)
                for k in ("euribor_3m", "overnight_households", "term_households_new")}
        mkt = long["euribor_3m"]
        # the balance sheet uses betas estimated since 2015; the full sample since 2003 is a robustness check
        series = {"Retail sight deposits (transactional, NMD)": "overnight_households",
                  "Retail savings deposits (non-transactional, NMD)": "term_households_new"}
        pairs = {name: (long[k][long[k].index >= "2015-01"], mkt[mkt.index >= "2015-01"])
                 for name, k in series.items()}
        rows = {}
        for k in ("overnight_households", "term_households_new"):
            for start in ("2003-01", "2015-01"):
                e = estimate_deposit_beta(long[k][long[k].index >= start], mkt[mkt.index >= start])
                rows[(k, f"{start} to {long[k].index[-1]}")] = e
        by_sample = pd.DataFrame(rows).T[["observations", "beta_long_run", "eg_t_stat", "eg_crit_5pct",
                                          "cointegrated_5pct", "half_life_raise_months", "half_life_cut_months",
                                          "t_stat_asymmetry"]]
        print(f"Data: ECB MIR statistics vs 3M Euribor, monthly, euro area, to {mkt.index[-1]}")
        print("Cointegration and pass-through by sample:")
        print(by_sample.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        pairs = {"Retail sight deposits (transactional, NMD)": simulate_deposit_rates(beta=0.15, seed=1),
                 "Retail savings deposits (non-transactional, NMD)": simulate_deposit_rates(beta=0.45, seed=2)}
        print("Data: SIMULATED series (offline demo). Run with --ecb for real ECB statistics.")
    est = {name: estimate_deposit_beta(d, m) for name, (d, m) in pairs.items()}
    est_tab = pd.DataFrame(est).T[["beta_long_run", "eg_t_stat", "eg_crit_5pct", "cointegrated_5pct",
                                   "beta_same_month", "adjustment_speed", "half_life_months",
                                   "half_life_raise_months", "half_life_cut_months", "t_stat_asymmetry",
                                   "cycle_beta_trough_to_peak", "cycle_beta_peak_to_trough", "r2_long_run"]]
    print(est_tab.T.to_string(float_format=lambda x: f"{x:.3f}"))
    for name, e in est.items():
        if not e["cointegrated_5pct"]:
            print(f"Warning: no cointegration at 5% for '{name}' (EG t = {e['eg_t_stat']:.2f} vs "
                  f"{e['eg_crit_5pct']:.2f}); the long-run beta may be spurious.")
    for pos in bs.positions:  # feed the estimated betas into the balance sheet
        if pos.name in est:
            pos.beta = float(np.clip(est[pos.name]["beta_long_run"], 0.0, 1.0))
    print("Estimated long-run betas are used for the NMD positions in the NII simulation below.")

    # ---------------------------------------------------------- balance sheet
    tot = bs.totals()
    section("Balance sheet (EUR m, illustrative)")
    print(bs.to_frame()[["name", "side", "notional", "rate", "maturity", "avg_life", "fixed", "cpr",
                         "core_share", "beta"]].to_string(
        index=False, na_rep="-", formatters={"rate": lambda x: f"{x:.2%}"}))
    print(f"\nAssets {tot['assets']:,.0f} | Liabilities {tot['liabilities']:,.0f} | "
          f"Book equity {tot['equity']:,.0f} | Tier 1 (assumed) {TIER1:,.0f}")

    section("Repricing gap (EUR m, assets - liabilities)")
    gap = bs.repricing_gap()
    print(gap)

    pv = present_value(bs, curve)
    section("Present value and DV01 by position (EUR m)")
    print(pv[["name", "side", "pv", "signed_dv01"]].to_string(index=False, formatters={"signed_dv01": "{:,.3f}".format}))
    krd = key_rate_dv01(bs, curve)
    print(f"\nNet DV01: {pv['signed_dv01'].sum():,.3f} EUR m per +1bp parallel")
    print("Key-rate DV01 (EUR m per +1bp at one tenor):")
    print(krd.rename(lambda k: f"{k:g}y").round(3).to_frame().T.to_string(index=False))

    # ------------------------------------------------------------------ IRRBB
    rep = irrbb_report(bs, curve, TIER1)
    section("IRRBB: ΔEVE under the six BCBS shocks (EUR; post-shock floor; behavioural prepayment)")
    t = rep["eve_table"].copy()
    t["% of Tier 1"] = t["ΔEVE"] / TIER1 * 100
    t["ΔEVE, static prepayment"] = eve_sensitivity(bs, curve, behavioural_prepayment=False)["ΔEVE"]
    print(t)
    print(f"\nWorst scenario: {rep['worst_scenario']} | ΔEVE {rep['worst_ΔEVE']:,.1f} "
          f"= {rep['ΔEVE_pct_tier1']:.1%} of Tier 1 | EVE outlier (< -15%): {rep['SOT_outlier_15pct']}")

    section("IRRBB: ΔNII, 12-month horizon, constant balance sheet, with deposit betas (deposit rates floored at 0%)")
    print(rep["nii_table"])
    print(f"Worst parallel ΔNII {rep['worst_ΔNII']:,.1f} = {rep['ΔNII_pct_tier1']:.1%} of Tier 1 | "
          f"NII outlier (< -5%): {rep['SOT_NII_outlier_5pct']}")
    zero_beta = sample_bank()
    for pos in zero_beta.positions:
        if pos.is_nmd:
            pos.beta = 0.0
    up0 = nii_sensitivity(zero_beta, curve).loc["parallel_up", "ΔNII"]
    print(f"For comparison, ΔNII in parallel_up if deposit betas were zero: {up0:,.1f} "
          f"(vs {rep['nii_table'].loc['parallel_up', 'ΔNII']:,.1f} with estimated betas)")

    # ---------------------------------------------------------------- hedging
    section("Hedging 1: smallest single pay-fixed swap that passes the EVE outlier test")
    hedges = []
    for tenor in (5.0, 7.0, 10.0, 15.0):
        h = size_hedge(bs, curve, TIER1, tenor=tenor, target=0.15)
        hedges.append(h)
        status = "passes" if h["target_met"] else "CANNOT pass at any size (best effort shown)"
        print(f"{tenor:>4.0f}y @ {h['rate']:.2%}: notional {h['notional']:>7,.0f} -> worst ΔEVE "
              f"{h['worst_ΔEVE']:>7,.1f} ({h['worst_ΔEVE'] / TIER1:.1%} of Tier 1), {status}")

    section("Hedging 2: cheapest swap portfolio keeping ΔEVE within 10% of Tier 1 in all scenarios (LP)")
    opt = optimise_hedge(bs, curve, TIER1, limit=0.10)
    print("Notional by tenor (EUR m; + pay fixed, - receive fixed):")
    print(opt["notionals"].rename(lambda k: f"{k:g}y").to_frame().T.to_string(index=False))
    hedged = apply_hedge(bs, curve, opt["notionals"])
    comp = pd.DataFrame({"ΔEVE unhedged": rep["eve_table"]["ΔEVE"], "ΔEVE hedged": eve_sensitivity(hedged, curve)["ΔEVE"]})
    print(comp)
    nii_h = nii_sensitivity(hedged, curve)
    print(f"Worst ΔEVE after hedge: {opt['worst_ΔEVE']:,.1f} ({opt['worst_ΔEVE'] / TIER1:.1%} of Tier 1), "
          f"gross notional {opt['gross_notional']:,.0f}")
    print("EVE vs NII trade-off, ΔNII parallel_up / parallel_down: "
          f"unhedged {rep['nii_table'].loc['parallel_up', 'ΔNII']:,.1f} / {rep['nii_table'].loc['parallel_down', 'ΔNII']:,.1f}"
          f" -> hedged {nii_h.loc['parallel_up', 'ΔNII']:,.1f} / {nii_h.loc['parallel_down', 'ΔNII']:,.1f}")

    # -------------------------------------------------------------- liquidity
    section("Liquidity Coverage Ratio (simplified)")
    for k, v in lcr(bs).items():
        print(f"{k:>18}: {v:,.2f}" if isinstance(v, float) else f"{k:>18}: {v}")
    section("Deposit-run survival horizon")
    for mult in (1.0, 2.0, 3.0):
        path = survival_horizon(bs, mult)
        survived = path["buffer"].iloc[-1] >= 0
        print(f"Run-off x{mult:.0f}: survives {'>' if survived else ''}{len(path)} days")

    # ----------------------------------------------------------------- charts
    # palette "Quiet Analytics", shared with the author's other projects
    navy, steel, stone, warm, charcoal = "#101828", "#5B7C99", "#A3A3A3", "#FAFAF7", "#111111"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": warm, "axes.facecolor": warm, "savefig.facecolor": warm,
        "axes.edgecolor": stone, "axes.labelcolor": charcoal, "text.color": charcoal,
        "xtick.color": "#555555", "ytick.color": "#555555", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "regular", "axes.titlesize": 13,
        "axes.titlelocation": "left", "grid.color": stone, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })
    labels = {"parallel_up": "Parallel up", "parallel_down": "Parallel down", "steepener": "Steepener",
              "flattener": "Flattener", "short_up": "Short up", "short_down": "Short down"}

    tt = np.linspace(0.1, 30, 300)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    shades = {"parallel_up": navy, "parallel_down": navy, "steepener": steel, "flattener": steel,
              "short_up": stone, "short_down": stone}
    for sc in SCENARIOS:
        y = (curve.zero_rate(tt) + floored_shift(curve, shock_curve(sc))(tt)) * 100
        ax.plot(tt, y, lw=1.0, color=shades[sc], ls="-" if sc.endswith("up") or sc == "steepener" else "--")
        # short-rate shocks fade out at the long end, so label them at the short end
        i, dx, ha = (0, -4, "right") if sc.startswith("short") else (-1, 4, "left")
        ax.annotate(labels[sc], (tt[i], y[i]), xytext=(dx, 0), textcoords="offset points", fontsize=8,
                    va="center", ha=ha, color=shades[sc])
    ax.plot(tt, curve.zero_rate(tt) * 100, color=charcoal, lw=2.2)
    ax.annotate("Base", (tt[-1], curve.zero_rate(tt[-1:])[0] * 100), xytext=(4, 0), textcoords="offset points",
                fontsize=8, va="center", fontweight="bold")
    ax.scatter(market.index, market.values * 100, color=charcoal, s=14, zorder=3)
    ax.set(xlabel="Maturity, years", ylabel="Zero rate, %", xlim=(-3.2, 34))
    ax.set_title("Base curve and the six BCBS shock scenarios (EUR, post-shock floor)", pad=12)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "curve_scenarios.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xpos = np.arange(len(comp))
    ax.bar(xpos - 0.2, comp["ΔEVE unhedged"], 0.38, color=navy, label="Unhedged")
    ax.bar(xpos + 0.2, comp["ΔEVE hedged"], 0.38, color=steel, label="Hedged (LP swap portfolio)")
    ax.axhline(-0.15 * TIER1, color=charcoal, ls="--", lw=1)
    ax.annotate("Supervisory outlier test: −15% of Tier 1", (len(comp) - 0.5, -0.15 * TIER1), xytext=(0, -12),
                textcoords="offset points", ha="right", fontsize=9)
    ax.axhline(0, color=stone, lw=0.8)
    ax.set_xticks(xpos, [labels[i] for i in comp.index])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("ΔEVE, EUR m")
    ax.set_title("Change in economic value of equity under the BCBS shocks", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_eve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    krd_h = key_rate_dv01(hedged, curve)
    xpos = np.arange(len(krd))
    ax.bar(xpos - 0.2, krd.values * 1000, 0.38, color=navy, label="Unhedged")
    ax.bar(xpos + 0.2, krd_h.values * 1000, 0.38, color=steel, label="Hedged")
    ax.set_xticks(xpos, [f"{k:g}y" for k in krd.index])
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=stone, lw=0.8)
    ax.set_ylabel("EUR thousand per +1bp")
    ax.set_title("Key-rate DV01 of economic value: where the rate risk sits on the curve", pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "key_rate_dv01.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(gap.index.astype(str), gap["gap"], color=[steel if v >= 0 else navy for v in gap["gap"]], width=0.6,
           label="Gap in bucket")
    ax.plot(gap.index.astype(str), gap["cumulative_gap"], color=charcoal, marker="o", ms=4, lw=1.2,
            label="Cumulative gap")
    ax.axhline(0, color=stone, lw=0.8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("EUR m, assets − liabilities")
    ax.set_title("Repricing gap by time bucket", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "repricing_gap.png", dpi=150)

    # ------------------------------------------- numbers quoted in the README
    t_txt = _fmt(t, {"% of Tier 1": "{:.1f}%"})
    nii_txt = _fmt(rep["nii_table"], {})
    beta_txt = est_tab.T.to_string(float_format=lambda x: f"{x:.3f}")
    hedge_txt = "\n".join(
        f"{h['tenor']:>4.0f}y @ {h['rate']:.2%}: notional {h['notional']:>6,.0f}, worst ΔEVE {h['worst_ΔEVE']:,.1f} "
        f"({h['worst_ΔEVE'] / TIER1:.1%} of Tier 1)" for h in hedges)
    lcr_res = lcr(bs)
    surv = {m: survival_horizon(bs, m) for m in (1.0, 2.0, 3.0)}
    surv_txt = ", ".join(f"run-off x{m:.0f}: {'>' if p_['buffer'].iloc[-1] >= 0 else ''}{len(p_)} days"
                         for m, p_ in surv.items())
    sample_txt = ("" if by_sample is None else
                  "### By sample (ECB data)\n\n```\n" + by_sample.to_string(float_format=lambda x: f"{x:.3f}")
                  + "\n```\n\n")
    data_note = (f"ECB AAA euro-area curve{' on ' + args.date if args.date else ' (latest)'}; ECB MIR deposit rates "
                 f"vs 3M Euribor to {mkt.index[-1]}" if args.ecb else
                 "illustrative curve and SIMULATED deposit-rate series (offline, not market data)")
    nt = rep["nii_table"]["ΔNII"]
    (out_dir / "summary.md").write_text(
        f"# Treasury ALM / IRRBB report\n\nData: {data_note}\n\n"
        f"Balance sheet (EUR m): assets {tot['assets']:,.0f}, liabilities {tot['liabilities']:,.0f}, "
        f"book equity {tot['equity']:,.0f}, Tier 1 (assumed) {TIER1:,.0f}\n\n"
        f"Yield curve: NSS fit RMSE {rmse:.2f} bp\n\n"
        f"## ΔEVE, six BCBS shocks (EUR m, post-shock floor)\n\n```\n{t_txt}\n```\n\n"
        f"Worst scenario: {rep['worst_scenario']}, ΔEVE {rep['worst_ΔEVE']:,.1f} = {rep['ΔEVE_pct_tier1']:.1%} of "
        f"Tier 1. Outlier (< −15%): {rep['SOT_outlier_15pct']}\n\n"
        f"Net DV01 {pv['signed_dv01'].sum() * 1000:,.0f} EUR thousand per +1bp. Key-rate DV01 (EUR thousand per +1bp): "
        + ", ".join(f"{k:g}y {v * 1000:,.0f}" for k, v in krd.items()) + "\n\n"
        f"## ΔNII, 12 months, constant balance sheet (EUR m, deposit rates floored at 0%)\n\n```\n{nii_txt}\n```\n\n"
        f"Worst parallel ΔNII {rep['worst_ΔNII']:,.1f} = {rep['ΔNII_pct_tier1']:.1%} of Tier 1. "
        f"Outlier (< −5%): {rep['SOT_NII_outlier_5pct']}\n\n"
        f"ΔNII parallel_up with zero deposit betas: {up0:,.1f} (vs {nt['parallel_up']:,.1f} with estimated betas)\n\n"
        f"## Deposit betas\n\n```\n{beta_txt}\n```\n\n{sample_txt}"
        f"## Hedging\n\nSmallest single pay-fixed swap that passes the 15% test:\n\n```\n{hedge_txt}\n```\n\n"
        f"Cheapest swap portfolio keeping ΔEVE within 10% of Tier 1 (LP), EUR m (+ pay fixed): "
        + ", ".join(f"{k:g}y {v:,.1f}" for k, v in opt["notionals"].items())
        + f". Gross notional {opt['gross_notional']:,.1f}; worst ΔEVE after hedge {opt['worst_ΔEVE']:,.1f} "
        f"({opt['worst_ΔEVE'] / TIER1:.1%} of Tier 1)\n\n```\n{_fmt(comp, {})}\n```\n\n"
        f"EVE vs NII: ΔNII parallel_up / parallel_down unhedged {nt['parallel_up']:,.1f} / {nt['parallel_down']:,.1f}, "
        f"hedged {nii_h.loc['parallel_up', 'ΔNII']:,.1f} / {nii_h.loc['parallel_down', 'ΔNII']:,.1f}\n\n"
        f"## Liquidity\n\nLCR {lcr_res['LCR']:.0%} (HQLA {lcr_res['HQLA']:,.0f}, net outflows "
        f"{lcr_res['net_outflows']:,.0f}). Survival horizon: {surv_txt}\n")

    comp.to_csv(out_dir / "delta_eve.csv")
    rep["nii_table"].to_csv(out_dir / "delta_nii.csv")
    gap.to_csv(out_dir / "repricing_gap.csv")
    krd.to_csv(out_dir / "key_rate_dv01.csv")
    est_tab.to_csv(out_dir / "deposit_betas.csv")
    if by_sample is not None:
        by_sample.to_csv(out_dir / "deposit_betas_by_sample.csv")
    print(f"\nCharts and tables saved to {out_dir}/")


if __name__ == "__main__":
    main()

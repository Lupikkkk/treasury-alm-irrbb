"""Interactive dashboard: `streamlit run app.py`"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alm import fit_nss, lcr, sample_bank, shock_curve, survival_horizon
from alm.data import fetch_ecb_curve, sample_curve
from alm.irrbb import (SCENARIOS, add_swap, eve_sensitivity, floored_shift, key_rate_dv01, nii_sensitivity,
                       optimise_hedge)

NAVY, STEEL, STONE = "#101828", "#5B7C99", "#A3A3A3"
LABELS = {"parallel_up": "Parallel up", "parallel_down": "Parallel down", "steepener": "Steepener",
          "flattener": "Flattener", "short_up": "Short up", "short_down": "Short down"}

st.set_page_config(page_title="Treasury ALM & IRRBB", layout="wide")
st.title("Treasury ALM & IRRBB")
st.caption("NSS curve · BCBS IRRBB shocks · prepayment and deposit-beta behaviour · key-rate DV01 · swap hedging · LCR")

with st.sidebar:
    use_ecb = st.checkbox("Use latest ECB AAA curve (internet)", value=False)
    tier1 = st.number_input("Tier 1 capital (EUR m)", value=550.0, step=50.0)
    st.subheader("Hedge")
    tenor = st.selectbox("Payer swap tenor (years)", [5, 7, 10, 15], index=2)
    notional = st.slider("Swap notional (EUR m)", 0, 4000, 0, step=50)
    st.subheader("Liquidity stress")
    mult = st.slider("Deposit run-off multiplier", 1.0, 5.0, 1.0, 0.5)


@st.cache_data
def get_curve(use_ecb):
    m = fetch_ecb_curve() if use_ecb else sample_curve()
    c, rmse = fit_nss(m.index.values, m.values)
    return m, c.params(), rmse


market, params, rmse = get_curve(use_ecb)
from alm.curve import NelsonSiegelSvensson

curve = NelsonSiegelSvensson(**params)
bs = sample_bank()
hedged = add_swap(bs, notional, tenor, curve=curve) if notional else bs

eve_h = eve_sensitivity(hedged, curve)["ΔEVE"]
nii_h = nii_sensitivity(hedged, curve)["ΔNII"]
c = st.columns(4)
c[0].metric("Worst ΔEVE, % of Tier 1", f"{eve_h.min() / tier1:.1%}", help="Supervisory outlier test: −15%")
c[1].metric("Worst scenario", LABELS[eve_h.idxmin()])
c[2].metric("Worst parallel ΔNII, EUR m", f"{nii_h[['parallel_up', 'parallel_down']].min():,.1f}")
c[3].metric("LCR", f"{lcr(bs, mult)['LCR']:.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Yield curve", "IRRBB: EVE", "IRRBB: NII", "Liquidity"])

with tab1:
    tt = np.linspace(0.1, 30, 300)
    fig = go.Figure()
    shades = {"parallel": NAVY, "steepener": STEEL, "flattener": STEEL, "short": STONE}
    for sc in SCENARIOS:
        fig.add_scatter(x=tt, y=(curve.zero_rate(tt) + floored_shift(curve, shock_curve(sc))(tt)) * 100,
                        name=LABELS[sc], line=dict(color=shades[sc.split("_")[0]], width=1.2,
                                                   dash="solid" if sc in ("parallel_up", "steepener", "short_up") else "dash"))
    fig.add_scatter(x=tt, y=curve.zero_rate(tt) * 100, name="Base (NSS)", line=dict(color="#111111", width=3))
    fig.add_scatter(x=market.index, y=market.values * 100, mode="markers", name="Input", marker=dict(color="#111111"))
    fig.update_layout(height=450, xaxis_title="Maturity (y)", yaxis_title="Zero rate (%)")
    st.plotly_chart(fig, width="stretch")
    st.write(f"NSS fit RMSE: **{rmse:.2f} bp**" + ("" if use_ecb else " (illustrative curve, not market data)"))
    st.dataframe(bs.to_frame())

with tab2:
    base_tab = eve_sensitivity(bs, curve)
    hedged_tab = eve_sensitivity(hedged, curve)
    comp = pd.DataFrame({"ΔEVE unhedged": base_tab["ΔEVE"], "ΔEVE hedged": hedged_tab["ΔEVE"]})
    comp["% Tier 1 (hedged)"] = comp["ΔEVE hedged"] / tier1
    fig = go.Figure()
    names = [LABELS[i] for i in comp.index]
    fig.add_bar(x=names, y=comp["ΔEVE unhedged"], name="Unhedged", marker_color=NAVY)
    fig.add_bar(x=names, y=comp["ΔEVE hedged"], name=f"With {notional:,} {tenor}y payer swap", marker_color=STEEL)
    fig.add_hline(y=-0.15 * tier1, line_dash="dash", line_color="#111111", annotation_text="Outlier test (−15% of Tier 1)")
    fig.update_layout(barmode="group", height=420, yaxis_title="EUR m")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(comp.style.format({"ΔEVE unhedged": "{:,.1f}", "ΔEVE hedged": "{:,.1f}", "% Tier 1 (hedged)": "{:.1%}"}))
    worst = comp["ΔEVE hedged"].min()
    (st.error if worst < -0.15 * tier1 else st.success)(
        f"Worst ΔEVE: {worst:,.1f} EUR m = {worst / tier1:.1%} of Tier 1")
    k0, k1 = key_rate_dv01(bs, curve), key_rate_dv01(hedged, curve)
    fig = go.Figure()
    fig.add_bar(x=[f"{k:g}y" for k in k0.index], y=k0.values * 1000, name="Unhedged", marker_color=NAVY)
    fig.add_bar(x=[f"{k:g}y" for k in k1.index], y=k1.values * 1000, name="Hedged", marker_color=STEEL)
    fig.update_layout(barmode="group", height=350, title="Key-rate DV01 (EUR thousand per +1bp)")
    st.plotly_chart(fig, width="stretch")
    limit = st.slider("Risk limit for the optimiser (% of Tier 1)", 5, 15, 10) / 100
    opt = optimise_hedge(bs, curve, tier1, limit=limit)
    st.write("Cheapest swap portfolio that meets the limit (EUR m; + pay fixed, - receive fixed):")
    st.dataframe(opt["notionals"].rename(lambda k: f"{k:g}y").to_frame().T)

with tab3:
    st.dataframe(pd.DataFrame({"ΔNII unhedged": nii_sensitivity(bs, curve)["ΔNII"],
                               "ΔNII hedged": nii_h}).style.format("{:,.1f}"))
    st.caption("12-month horizon, constant balance sheet, deposit rates floored at 0%. A payer swap reduces EVE risk but changes NII sensitivity, "
               "which is the classic EVE vs. NII trade-off in ALM.")

with tab4:
    l = lcr(bs, mult)
    c = st.columns(3)
    c[0].metric("LCR", f"{l['LCR']:.0%}")
    c[1].metric("HQLA", f"{l['HQLA']:,.0f}")
    c[2].metric("Net 30d outflows", f"{l['net_outflows']:,.0f}")
    path = survival_horizon(bs, mult)
    fig = go.Figure(go.Scatter(x=path.index, y=path["buffer"], fill="tozeroy", line=dict(color=STEEL)))
    fig.update_layout(title="Liquidity buffer under deposit run (EUR m)", xaxis_title="Day", height=380)
    st.plotly_chart(fig, width="stretch")

"""Yield-curve input data.

``fetch_ecb_curve`` downloads euro-area AAA government spot rates from the ECB Data Portal
(series YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_<tenor>). ``sample_curve`` returns an illustrative
upward-sloping curve so the project runs offline. The sample values are made up for
demonstration and are NOT market data.
"""

from __future__ import annotations

import io

import pandas as pd

TENORS = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "15Y": 15,
          "20Y": 20, "30Y": 30}


def sample_curve() -> pd.Series:
    """Illustrative zero curve (decimals), indexed by maturity in years. Not market data."""
    vals = [0.0205, 0.0201, 0.0198, 0.0203, 0.0212, 0.0233, 0.0254, 0.0281, 0.0306, 0.0316, 0.0318]
    return pd.Series(vals, index=list(TENORS.values()), name="zero_rate")


def fetch_ecb_curve(date: str | None = None) -> pd.Series:
    """Download the ECB AAA euro-area spot curve for ``date`` (YYYY-MM-DD) or the latest date.

    Requires internet access and ``requests``.
    """
    import requests

    out = {}
    for label, t in TENORS.items():
        key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{label}"
        url = f"https://data-api.ecb.europa.eu/service/data/YC/{key}?format=csvdata"
        if date:
            url += f"&startPeriod={date}&endPeriod={date}"
        else:
            url += "&lastNObservations=1"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        if not r.text.strip():
            raise ValueError(f"ECB has no yield-curve observation for {date} (weekend, holiday or not yet "
                             "published); pick another date or omit --date for the latest curve")
        df = pd.read_csv(io.StringIO(r.text))
        out[t] = float(df["OBS_VALUE"].iloc[-1]) / 100.0
    return pd.Series(out, name="zero_rate").sort_index()

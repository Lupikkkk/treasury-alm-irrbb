# Treasury ALM / IRRBB report

Data: illustrative curve and SIMULATED deposit-rate series (offline, not market data)

Balance sheet (EUR m): assets 7,100, liabilities 6,500, book equity 600, Tier 1 (assumed) 550

Yield curve: NSS fit RMSE 0.78 bp

## ΔEVE, six BCBS shocks (EUR m, post-shock floor)

```
                   EVE    ΔEVE % of Tier 1 ΔEVE, static prepayment
scenario                                                          
parallel_up      920.9  -117.2      -21.3%                  -108.7
parallel_down  1,169.6   131.6       23.9%                   155.5
steepener        972.9   -65.2      -11.9%                   -63.2
flattener      1,077.3    39.2        7.1%                    50.1
short_up       1,053.2    15.1        2.7%                     8.9
short_down     1,024.0   -14.0       -2.6%                    -9.1
```

Worst scenario: parallel_up, ΔEVE -117.2 = -21.3% of Tier 1. Outlier (< −15%): True

Net DV01 -647 EUR thousand per +1bp. Key-rate DV01 (EUR thousand per +1bp): 0.25y -4, 1y 91, 2y -11, 3y -79, 5y -89, 7y 428, 10y -336, 15y -465, 20y -183, 30y 0

## ΔNII, 12 months, constant balance sheet (EUR m, deposit rates floored at 0%)

```
                 NII   ΔNII
scenario                   
parallel_up    187.2   10.4
parallel_down  165.2  -11.7
short_up       185.1    8.2
short_down     166.5  -10.3
```

Worst parallel ΔNII -11.7 = -2.1% of Tier 1. Outlier (< −5%): False

ΔNII parallel_up with zero deposit betas: 25.0 (vs 10.4 with estimated betas)

## Deposit betas

```
                          Retail sight deposits (transactional, NMD) Retail savings deposits (non-transactional, NMD)
beta_long_run                                                  0.141                                            0.422
eg_t_stat                                                     -2.755                                           -2.615
eg_crit_5pct                                                  -3.388                                           -3.388
cointegrated_5pct                                              False                                            False
beta_same_month                                                0.038                                            0.045
adjustment_speed                                              -0.268                                           -0.152
half_life_months                                               2.221                                            4.190
half_life_raise_months                                         1.867                                            2.892
half_life_cut_months                                           2.740                                            5.811
t_stat_asymmetry                                               0.585                                            2.705
cycle_beta_trough_to_peak                                      0.143                                            0.403
cycle_beta_peak_to_trough                                      0.106                                            0.234
r2_long_run                                                    0.967                                            0.924
```

## Hedging

Smallest single pay-fixed swap that passes the 15% test:

```
   5y @ 2.35%: notional    381, worst ΔEVE -82.5 (-15.0% of Tier 1)
   7y @ 2.56%: notional    286, worst ΔEVE -82.5 (-15.0% of Tier 1)
  10y @ 2.80%: notional    215, worst ΔEVE -82.5 (-15.0% of Tier 1)
  15y @ 3.03%: notional    163, worst ΔEVE -82.5 (-15.0% of Tier 1)
```

Cheapest swap portfolio keeping ΔEVE within 10% of Tier 1 (LP), EUR m (+ pay fixed): 2y 0.0, 5y 0.0, 7y 0.0, 10y 0.0, 15y 0.0, 20y 245.8. Gross notional 245.8; worst ΔEVE after hedge -55.0 (-10.0% of Tier 1)

```
              ΔEVE unhedged ΔEVE hedged
scenario                               
parallel_up          -117.2       -55.0
parallel_down         131.6        42.6
steepener             -65.2       -37.2
flattener              39.2        18.9
short_up               15.1        17.9
short_down            -14.0       -16.9
```

EVE vs NII: ΔNII parallel_up / parallel_down unhedged 10.4 / -11.7, hedged 14.1 / -15.4

## Liquidity

LCR 320% (HQLA 1,755, net outflows 549). Survival horizon: run-off x1: 109 days, run-off x2: 50 days, run-off x3: 29 days

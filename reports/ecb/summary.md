# Treasury ALM / IRRBB report

Data: ECB AAA euro-area curve on 2026-11-12; ECB MIR deposit rates vs 3M Euribor to 2026-09

Balance sheet (EUR m): assets 7,100, liabilities 6,500, book equity 600, Tier 1 (assumed) 550

Yield curve: NSS fit RMSE 0.00 bp

## ΔEVE, six BCBS shocks (EUR m, post-shock floor)

```
                   EVE    ΔEVE % of Tier 1 ΔEVE, static prepayment
scenario                                                          
parallel_up      899.2  -110.4      -20.1%                   -99.4
parallel_down  1,133.2   123.6       22.5%                   142.6
steepener        945.8   -63.8      -11.6%                   -58.7
flattener      1,049.3    39.7        7.2%                    46.7
short_up       1,021.1    11.6        2.1%                     8.9
short_down       998.9   -10.6       -1.9%                    -9.2
```

Worst scenario: parallel_up, ΔEVE -110.4 = -20.1% of Tier 1. Outlier (< −15%): True

Net DV01 -593 EUR thousand per +1bp. Key-rate DV01 (EUR thousand per +1bp): 0.25y -4, 1y 91, 2y -11, 3y -76, 5y -86, 7y 409, 10y -316, 15y -432, 20y -167, 30y 0

## ΔNII, 12 months, constant balance sheet (EUR m, deposit rates floored at 0%)

```
                 NII   ΔNII
scenario                   
parallel_up    200.3    9.9
parallel_down  177.5  -12.8
short_up       198.0    7.7
short_down     178.2  -12.2
```

Worst parallel ΔNII -12.8 = -2.3% of Tier 1. Outlier (< −5%): False

ΔNII parallel_up with zero deposit betas: 25.0 (vs 9.9 with estimated betas)

## Deposit betas

```
                          Retail sight deposits (transactional, NMD) Retail savings deposits (non-transactional, NMD)
beta_long_run                                                  0.073                                            0.636
eg_t_stat                                                     -2.536                                           -3.606
eg_crit_5pct                                                  -3.380                                           -3.380
cointegrated_5pct                                              False                                             True
beta_same_month                                                0.018                                            0.485
adjustment_speed                                              -0.108                                           -0.133
half_life_months                                               6.078                                            4.839
half_life_raise_months                                         3.321                                            2.114
half_life_cut_months                                          16.928                                            8.857
t_stat_asymmetry                                               3.135                                            2.449
cycle_beta_trough_to_peak                                      0.077                                            0.698
cycle_beta_peak_to_trough                                      0.045                                            0.785
r2_long_run                                                    0.817                                            0.972
```

### By sample (ECB data)

```
                                        observations beta_long_run eg_t_stat eg_crit_5pct cointegrated_5pct half_life_raise_months half_life_cut_months t_stat_asymmetry
overnight_households 2003-01 to 2026-09          285         0.175    -0.977       -3.358             False                 47.328               28.529           -0.492
                     2015-01 to 2026-09          141         0.073    -2.536       -3.380             False                  3.321               16.928            3.135
term_households_new  2003-01 to 2026-09          285         0.645    -1.866       -3.358             False                 29.568              105.410            0.443
                     2015-01 to 2026-09          141         0.636    -3.606       -3.380              True                  2.114                8.857            2.449
```

## Hedging

Smallest single pay-fixed swap that passes the 15% test:

```
   5y @ 3.18%: notional    312, worst ΔEVE -82.5 (-15.0% of Tier 1)
   7y @ 3.26%: notional    234, worst ΔEVE -82.5 (-15.0% of Tier 1)
  10y @ 3.40%: notional    177, worst ΔEVE -82.5 (-15.0% of Tier 1)
  15y @ 3.57%: notional    135, worst ΔEVE -82.5 (-15.0% of Tier 1)
```

Cheapest swap portfolio keeping ΔEVE within 10% of Tier 1 (LP), EUR m (+ pay fixed): 2y 0.0, 5y 0.0, 7y 0.0, 10y 0.0, 15y 0.0, 20y 227.2. Gross notional 227.2; worst ΔEVE after hedge -55.0 (-10.0% of Tier 1)

```
              ΔEVE unhedged ΔEVE hedged
scenario                               
parallel_up          -110.4       -55.0
parallel_down         123.6        44.8
steepener             -63.8       -39.2
flattener              39.7        22.0
short_up               11.6        14.4
short_down            -10.6       -13.5
```

EVE vs NII: ΔNII parallel_up / parallel_down unhedged 9.9 / -12.8, hedged 13.3 / -16.3

## Liquidity

LCR 320% (HQLA 1,755, net outflows 549). Survival horizon: run-off x1: 109 days, run-off x2: 50 days, run-off x3: 29 days

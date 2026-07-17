# Biomarker Gran Prix — results summary

Runs surveyed: **10** · genome-wide hits (5e-8): **54** unique (run, predictor).

## Coverage & calibration (λ-GC per source)

| run | model | N | variants | GW hits | λ_AJ | λ_ALL | λ_EUR | λ_FE | λ_RE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analytes_vs_updrs | linear | 2000–2000 | 40 | 5 | — | 0.953 | — | 0.953 | 0.953 |
| cox_dementia | cox | 2000–2000 | 8,844 | 9 | — | 1.024 | — | 1.027 | 1.027 |
| gallop_bmkX | gallop | 2000–2000 | 8,844 | 1 | — | 0.968 | — | 0.968 | 0.968 |
| linear_bmkX | linear | 2000–2000 | 8,844 | 7 | — | 1.029 | — | 1.029 | 1.029 |
| linear_bmkX_conditional | linear | 2000–2000 | 8,843 | 6 | — | 1.008 | — | 1.008 | 1.008 |
| linear_bmkX_derivecond | linear | 2000–2000 | 8,843 | 6 | — | 1.008 | — | 1.008 | 1.008 |
| linear_bmkX_strat | linear | 618–1382 | 8,914 | 7 | 0.967 | — | 1.015 | 0.999 | 0.800 |
| logistic_PD | logistic | 2000–2000 | 8,844 | 7 | — | 1.040 | — | 1.040 | 1.040 |
| slope_bmkX | slope | 2000–2000 | 8,844 | 0 | — | 0.953 | — | 0.954 | 0.954 |
| trajgwas_bmkX | trajgwas | 2000–2000 | 8,841 | 6 | — | 0.984 | — | 0.986 | 0.986 |

> λ close to 1.0 = well-calibrated. FE/RE are the meta fixed/random-effects λ (RE is conservative at small #studies).

## λ-GC by MAF band (primary source)

| run | source | λ_all | λ_common | λ_low-freq | λ_rare |
| --- | --- | --- | --- | --- | --- |
| analytes_vs_updrs | FE | 0.953 | — | — | — |
| cox_dementia | FE | 1.027 | 1.012 | 1.168 | — |
| gallop_bmkX | FE | 0.968 | 0.960 | 1.045 | — |
| linear_bmkX | FE | 1.029 | 1.009 | 1.121 | — |
| linear_bmkX_conditional | FE | 1.008 | 1.002 | 1.038 | — |
| linear_bmkX_derivecond | FE | 1.008 | 1.002 | 1.038 | — |
| linear_bmkX_strat | FE | 0.999 | 0.996 | 1.019 | — |
| logistic_PD | FE | 1.040 | 1.040 | 1.033 | — |
| slope_bmkX | FE | 0.954 | 0.995 | 0.771 | — |
| trajgwas_bmkX | FE | 0.986 | 1.002 | 0.889 | — |

> Rare/low-freq inflation above the common band flags residual stratification or score-test miscalibration in that band.

## Genome-wide-significant hits (deduped, sorted by P)

| run | predictor | CHR | POS | source | beta | P |
| --- | --- | --- | --- | --- | --- | --- |
| analytes_vs_updrs | PRTN0031_NPX | — | — | FE | -2.3 | 2.76e-71 |
| analytes_vs_updrs | PRTN0019_NPX | — | — | FE | -2.01 | 7.42e-53 |
| analytes_vs_updrs | PRTN0003_NPX | — | — | FE | +1.79 | 2.47e-41 |
| cox_dementia | rs11192340 | 4 | 153,268,344 | FE | +0.568 | 2.45e-26 |
| cox_dementia | rs18024796 | 10 | 212,690,954 | FE | +0.539 | 6.99e-23 |
| trajgwas_bmkX | rs96879083 | 22 | 52,940,321 | FE | +1.49 | 9.80e-19 |
| analytes_vs_updrs | PRTN0018_NPX | — | — | FE | -1.2 | 1.16e-18 |
| analytes_vs_updrs | PRTN0033_NPX | — | — | FE | +1.18 | 3.96e-18 |
| cox_dementia | rs39074795 | 15 | 150,089,482 | FE | -0.457 | 8.05e-18 |
| cox_dementia | rs10219394 | 9 | 246,856,212 | FE | +0.454 | 1.96e-16 |
| trajgwas_bmkX | rs45732950 | 18 | 221,755,405 | FE | +1.12 | 3.85e-16 |
| logistic_PD | rs21176249 | 8 | 212,775,522 | FE | +0.642 | 5.94e-16 |
| linear_bmkX_conditional | rs3112342 | 14 | 128,789,153 | FE | +0.532 | 1.53e-15 |
| linear_bmkX_derivecond | rs3112342 | 14 | 128,789,153 | FE | +0.532 | 1.53e-15 |
| cox_dementia | rs30543286 | 4 | 214,434,268 | FE | -0.5 | 1.81e-15 |
| linear_bmkX_strat | rs3112342 | 14 | 128,789,153 | FE | +0.533 | 1.99e-15 |
| cox_dementia | rs83894229 | 10 | 140,296,894 | FE | +0.525 | 2.82e-15 |
| linear_bmkX | rs3112342 | 14 | 128,789,153 | FE | +0.528 | 4.07e-15 |
| trajgwas_bmkX | rs46326268 | 18 | 32,708,113 | FE | +1.46 | 7.42e-15 |
| trajgwas_bmkX | rs16399730 | 12 | 211,031,900 | FE | +1.17 | 3.77e-14 |
| cox_dementia | rs20892510 | 16 | 73,144,387 | ALL | +0.422 | 1.13e-13 |
| trajgwas_bmkX | rs7040632 | 5 | 73,099,526 | FE | +1.64 | 2.22e-13 |
| cox_dementia | rs39955811 | 1 | 71,300,449 | FE | -0.398 | 4.13e-13 |
| logistic_PD | rs52619689 | 12 | 188,705,424 | FE | +0.485 | 8.71e-13 |
| logistic_PD | rs44297183 | 9 | 48,091,960 | FE | -0.486 | 1.42e-12 |
| cox_dementia | rs56201475 | 4 | 178,159,063 | FE | -0.384 | 1.74e-12 |
| linear_bmkX | rs75165081 | 7 | 166,041,287 | FE | -0.531 | 6.47e-12 |
| logistic_PD | rs26065390 | 18 | 78,179,895 | FE | +0.457 | 7.00e-12 |
| linear_bmkX_strat | rs75165081 | 7 | 166,041,287 | FE | -0.523 | 1.58e-11 |
| logistic_PD | rs52170564 | 15 | 63,277,523 | FE | +0.482 | 3.33e-11 |
| linear_bmkX_derivecond | rs75165081 | 7 | 166,041,287 | FE | -0.508 | 3.99e-11 |
| linear_bmkX_conditional | rs75165081 | 7 | 166,041,287 | FE | -0.508 | 3.99e-11 |
| trajgwas_bmkX | rs57651983 | 4 | 45,009,348 | FE | +0.907 | 1.37e-10 |
| linear_bmkX_derivecond | rs92203642 | 4 | 101,244,187 | FE | -0.453 | 8.33e-10 |
| linear_bmkX_conditional | rs92203642 | 4 | 101,244,187 | FE | -0.453 | 8.33e-10 |
| linear_bmkX | rs92203642 | 4 | 101,244,187 | FE | -0.454 | 1.03e-09 |
| linear_bmkX_strat | rs76904798 | 12 | 40,340,000 | FE | +0.445 | 1.24e-09 |
| linear_bmkX | rs37486136 | 3 | 47,552,772 | FE | -0.399 | 1.59e-09 |
| linear_bmkX_derivecond | rs37486136 | 3 | 47,552,772 | FE | -0.393 | 1.93e-09 |
| linear_bmkX_conditional | rs37486136 | 3 | 47,552,772 | FE | -0.393 | 1.93e-09 |
| linear_bmkX_strat | rs92203642 | 4 | 101,244,187 | FE | -0.447 | 2.00e-09 |
| gallop_bmkX | rs95456322 | 2 | 228,962,432 | FE | -0.206 | 2.33e-09 |
| linear_bmkX | rs76904798 | 12 | 40,340,000 | FE | +0.434 | 3.44e-09 |
| linear_bmkX_strat | rs28509936 | 11 | 188,717,919 | EUR | +0.492 | 3.67e-09 |
| linear_bmkX_strat | rs37486136 | 3 | 47,552,772 | FE | -0.389 | 3.74e-09 |
| linear_bmkX_conditional | rs28509936 | 11 | 188,717,919 | FE | +0.393 | 3.96e-09 |
| linear_bmkX_derivecond | rs28509936 | 11 | 188,717,919 | FE | +0.393 | 3.96e-09 |
| linear_bmkX | rs28509936 | 11 | 188,717,919 | FE | +0.394 | 5.03e-09 |
| linear_bmkX_strat | rs13936515 | 9 | 62,425,882 | FE | -0.417 | 5.06e-09 |
| linear_bmkX | rs13936515 | 9 | 62,425,882 | FE | -0.409 | 9.87e-09 |

_…4 more hits omitted._


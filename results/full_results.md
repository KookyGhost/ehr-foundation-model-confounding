# Full reported result tables

These tables collect the reported results used in the confounding-adjustment study. The main script recomputes analysis outputs from source EHRSHOT data and the pretrained CLMBR checkpoint. The tables here are included as reference outputs for comparison.

The machine-readable results use one canonical CSV per analysis: `experiment1_clinical_only.csv`, `experiment2_temporal_only.csv`, `experiment3_mixed_clinical_temporal.csv`, and `real_metformin_vs_sulfonylurea.csv`. A tidy combined version is provided in `all_results_long.csv`.

## Experiment 1 — clinical-only confounding

| Method   |   OOF_AUC |   OOF_logloss |   Estimate_no_noise |   Bias_no_noise |   Bias_reduction_pct |
|:---------|----------:|--------------:|--------------------:|----------------:|---------------------:|
| Oracle   |    0.7952 |        0.5300 |             -4.9264 |          0.0736 |              97.5467 |
| Curated  |    0.7898 |        0.5353 |             -4.8834 |          0.1166 |              96.1133 |
| Count    |    0.5575 |        0.6702 |             -2.2066 |          2.7934 |               6.8867 |
| CLMBR    |    0.6530 |        0.6369 |             -2.5795 |          2.4205 |              19.3167 |
| Crude    |  nan      |      nan      |             -2.0000 |          3.0000 |               0.0000 |


### 500-outcome-simulation summary

| Method   |   Mean_estimate |   Mean_bias |   MC_SD |   RMSE |   Abs_bias |   Bias_reduction_vs_crude_pct |
|:---------|----------------:|------------:|--------:|-------:|-----------:|------------------------------:|
| Oracle   |          -4.928 |       0.072 |   0.105 |  0.127 |      0.072 |                        97.592 |
| Curated  |          -4.885 |       0.115 |   0.104 |  0.155 |      0.115 |                        96.176 |
| CLMBR    |          -2.581 |       2.419 |   0.095 |  2.421 |      2.419 |                        19.352 |
| Count    |          -2.207 |       2.793 |   0.095 |  2.795 |      2.793 |                         6.879 |
| Crude    |          -2.000 |       3.000 |   0.093 |  3.001 |      3.000 |                         0.000 |


## Experiment 2 — temporal-only confounding

### Treatment-assignment metrics

| Method            |   OOF_AUC |   OOF_logloss |
|:------------------|----------:|--------------:|
| Count             |    0.5325 |        0.6834 |
| Curated           |    0.5982 |        0.6568 |
| CLMBR             |    0.5948 |        0.6572 |
| Explicit Temporal |    0.7878 |        0.5335 |
| Oracle            |    0.8080 |        0.5065 |


### 500-outcome-simulation summary

| Method            |   Mean_estimate |   Mean_bias |   MC_SD |   RMSE |   Abs_bias |   Bias_reduction_vs_crude_pct |
|:------------------|----------------:|------------:|--------:|-------:|-----------:|------------------------------:|
| Explicit Temporal |          -4.974 |       0.026 |   0.104 |  0.107 |      0.026 |                        99.009 |
| Oracle            |          -4.963 |       0.037 |   0.107 |  0.113 |      0.037 |                        98.590 |
| CLMBR             |          -2.651 |       2.349 |   0.091 |  2.351 |      2.349 |                        10.480 |
| Curated           |          -2.470 |       2.530 |   0.092 |  2.531 |      2.530 |                         3.582 |
| Count             |          -2.435 |       2.565 |   0.091 |  2.566 |      2.565 |                         2.248 |
| Crude             |          -2.376 |       2.624 |   0.090 |  2.625 |      2.624 |                         0.000 |


## Experiment 3 — mixed clinical + temporal confounding

| Method              |   OOF_AUC |   OOF_logloss |   Estimate_no_noise |   Bias_no_noise |   Bias_reduction_pct |
|:--------------------|----------:|--------------:|--------------------:|----------------:|---------------------:|
| Count               |    0.5350 |        0.6770 |             -2.1170 |          2.8830 |               3.9000 |
| CLMBR               |    0.6400 |        0.6420 |             -2.4310 |          2.5690 |              14.4000 |
| Naive Hybrid        |    0.6980 |        0.6140 |             -3.1010 |          1.8990 |              36.7000 |
| Explicit Temporal   |    0.7040 |        0.6060 |             -3.2057 |          1.7943 |              40.1912 |
| Curated             |    0.7190 |        0.5940 |             -3.5329 |          1.4671 |              51.0967 |
| Offset Hybrid       |    0.7250 |        0.5900 |             -3.6060 |          1.3940 |              53.5000 |
| Clinical + Temporal |    0.7950 |        0.5290 |             -4.7280 |          0.2720 |              90.9331 |
| Oracle              |    0.8020 |        0.5210 |             -4.8835 |          0.1165 |              96.1162 |
| Crude               |  nan      |      nan      |             -2.0000 |          3.0000 |               0.0000 |


The AUC/log-loss columns above use the latest refit summarized in `figures/figure3_mixed_auc_logloss.png`. Crude has no propensity model, so AUC and log loss are not defined.


## Real Metformin vs sulfonylurea treatment assignment

| Method                 |   OOF_AUC |   OOF_logloss |   Delta_AUC_vs_Curated |   Logloss_improvement_vs_Curated_pct |
|:-----------------------|----------:|--------------:|-----------------------:|-------------------------------------:|
| Curated                |    0.6350 |        0.6064 |                 0.0000 |                               0.0000 |
| Curated + Temporal     |    0.6480 |        0.6067 |                 0.0130 |                              -0.0495 |
| CLMBR                  |    0.7510 |        0.5845 |                 0.1160 |                               3.6115 |
| Curated + CLMBR Offset |    0.7430 |        0.5478 |                 0.1080 |                               9.6636 |


This real-data table evaluates treatment-selection prediction only; it is not a known-truth causal-effect validation.

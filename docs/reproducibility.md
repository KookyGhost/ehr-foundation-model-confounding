# Reproducibility notes

## External inputs

A clean run requires:

1. Raw EHRSHOT MEDS parquet files.
2. The pretrained CLMBR-T-Base checkpoint.
3. Internet access to RxNav for the active-comparator drug vocabularies.

## No generated analysis inputs

The main script does not require pregenerated propensity scores, weights, Count matrices, CLMBR matrices, or result tables as inputs. It reconstructs those quantities during execution and writes outputs to the requested output directory.


## Cross-fitting

Propensity scores are evaluated out of fold. Hyperparameter selection is restricted to training data inside the cross-fitting workflow.

## Standardized mean differences

The current analysis uses the unweighted pre-adjustment pooled SD as the SMD denominator. This fixes the scale across weighting methods so post-weighting SMDs are directly comparable.

Because older balance tables were generated under a different denominator convention, this repository does not package those older balance values as reference results. Rerun the current script to obtain corrected balance tables.

## Mixed-test log loss

The latest Test 3 refit includes full out-of-fold probability vectors for Count, CLMBR, Naive Hybrid, Explicit Temporal, Curated, Offset Hybrid, Clinical + Temporal, and Oracle. This permits a complete log-loss comparison rather than leaving the high-dimensional methods as missing values.

## Reference tables

The CSV files under `results/` are provided as reported reference outputs. They are never read by the main analysis script to generate results.

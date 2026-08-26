# Results

This directory contains the reported summary tables for the confounding-adjustment study.

The four main analyses are represented symmetrically with one CSV each:

- `experiment1_clinical_only.csv` — clinical-only treatment-model metrics, no-noise effect recovery, and 500-simulation summary.
- `experiment2_temporal_only.csv` — temporal-only treatment-model metrics and 500-simulation effect-recovery summary.
- `experiment3_mixed_clinical_temporal.csv` — mixed-confounding AUC, log loss, no-noise effect estimate, bias, and bias reduction.
- `real_metformin_vs_sulfonylurea.csv` — real treatment-assignment AUC and log loss.

Supporting files:

- `all_results_long.csv` — tidy long-format master table containing all numeric results from the four analysis CSVs.
- `cohort_summary.csv` — key cohort sizes.
- `full_results.md` — human-readable combined tables.

The main analysis script writes regenerated result files when run. These packaged tables are reference outputs and are not loaded as analysis inputs.

Fixed-scale SMD/balance tables are intentionally not prepopulated here because the SMD denominator was corrected in the current analysis code. Rerun the script to generate balance tables under the corrected convention rather than mixing in older balance values.

# Stress-Testing EHR Foundation Models for Confounding Adjustment

This repository contains the **confounding-adjustment portion only** of a study evaluating whether a pretrained longitudinal EHR foundation model can add useful confounding information beyond conventional covariates in target trial emulation.

It intentionally excludes indication-expansion outcome analyses, AKI/anemia/dyspnea screening, scGPT, single-cell analyses, and genomic-distance analyses.

## Study question

**Can an EHR foundation model improve confounding adjustment, especially when the relevant longitudinal confounding structure is not known in advance?**

The analysis uses three controlled plasmode experiments plus a real treatment-assignment analysis:

1. **Clinical-only confounding:** treatment and outcome depend on SBP, DBP, BMI, and creatinine.
2. **Temporal-only confounding:** treatment and outcome depend on engineered longitudinal trend/acceleration features.
3. **Mixed clinical + temporal confounding:** both blocks contribute to treatment selection and outcome.
4. **Real Metformin vs sulfonylurea treatment assignment:** compares conventional, explicit temporal, CLMBR, and offset-hybrid propensity models when the true prescribing mechanism is unknown.

## Repository contents

```text
ehr-foundation-model-confounding/
├── ehr_foundation_model_confounding_reproducible.py
├── README.md
├── DATA_ACCESS.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
├── docs/
│   ├── analysis_overview.md
│   └── reproducibility.md
├── results/
│   ├── README.md
│   ├── full_results.md
│   ├── all_results_long.csv
│   ├── cohort_summary.csv
│   ├── experiment1_clinical_only.csv
│   ├── experiment2_temporal_only.csv
│   ├── experiment3_mixed_clinical_temporal.csv
│   └── real_metformin_vs_sulfonylurea.csv
├── figures/
│   └── figure3_mixed_auc_logloss.png
└── .github/workflows/
    └── syntax-check.yml
```

## Main analysis code

The main script is:

```text
ehr_foundation_model_confounding_reproducible.py
```

It rebuilds the common EHRSHOT cohort, engineered covariates, Count representation, CLMBR representations, treatment/outcome simulations, cross-fitted propensity scores, overlap weights, and analysis result tables.

Analysis-derived result files are **not required as inputs** for a clean run. The external inputs are the source EHRSHOT MEDS data and the pretrained CLMBR checkpoint.

## Key reported results

The complete reported summary tables are in [`results/full_results.md`](results/full_results.md). Each of the three plasmode experiments has one matching CSV, the real-data analysis has one CSV, and `all_results_long.csv` provides a single tidy master table.

The latest mixed-confounding treatment-model results are:

| Method | OOF AUC ↑ | OOF log loss ↓ | Estimated effect | Bias reduction |
|---|---:|---:|---:|---:|
| Count | 0.535 | 0.677 | -2.117 | 3.9% |
| CLMBR | 0.640 | 0.642 | -2.431 | 14.4% |
| Naive Hybrid | 0.698 | 0.614 | -3.101 | 36.7% |
| Explicit Temporal | 0.704 | 0.606 | -3.206 | 40.2% |
| Curated | 0.719 | 0.594 | -3.533 | 51.1% |
| Offset Hybrid | 0.725 | 0.590 | -3.606 | 53.5% |
| Clinical + Temporal | 0.795 | 0.529 | -4.728 | 90.9% |
| Oracle | 0.802 | 0.521 | -4.884 | 96.1% |

The true simulated treatment effect is **-5**.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python ehr_foundation_model_confounding_reproducible.py \
  --ehr-root /path/to/EHRSHOT \
  --clmbr-model-path /path/to/clmbr-t-base \
  --output-dir ./confounding_outputs
```

For Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%run ehr_foundation_model_confounding_reproducible.py \
    --ehr-root /content/drive/MyDrive/EHRSHOT
```

## Interpretation

The simulations support three main conclusions:

- When the true clinical confounders are known and observed directly, explicit clinical adjustment performs near the Oracle benchmark.
- When the true temporal confounders are explicitly engineered, conventional propensity adjustment again performs near the Oracle benchmark; CLMBR captures some, but not all, of that temporal signal.
- In mixed confounding, simply concatenating clinical variables with the high-dimensional CLMBR representation can degrade performance, while preserving the curated clinical model as an offset and using CLMBR as a residual correction produces a modest improvement.

In the real Metformin-vs-sulfonylurea treatment-assignment analysis, CLMBR captures substantially more treatment-selection information than the hand-engineered temporal block. Because causal ground truth is unavailable in real data, this is evidence about **treatment-selection information**, not direct proof of causal-bias reduction.

## Data/model licenses

This repository's code is licensed separately from EHRSHOT and CLMBR. The repository does not redistribute either external resource. See [`DATA_ACCESS.md`](DATA_ACCESS.md).

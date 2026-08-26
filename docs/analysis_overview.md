# Analysis overview

## Common plasmode cohort

All three controlled stress tests use the same 1,922-patient EHRSHOT cohort. Each patient has a 365-day pre-index history and at least 20 distinct dates of clinical activity in that year.

## Representations

- **Curated:** conventional hand-specified baseline summaries; in clinical/mixed experiments this also includes explicit clinical measurements, recency, and missingness indicators.
- **Count:** sparse counts of EHR event codes.
- **Explicit Temporal:** engineered longitudinal trend and acceleration summaries.
- **CLMBR:** pretrained longitudinal EHR embedding plus the representation-to-index gap anchor in the plasmode experiments.
- **Naive Hybrid:** direct concatenation of Curated and CLMBR.
- **Offset Hybrid:** Curated propensity logit preserved as an offset; CLMBR models residual treatment-selection information.
- **Oracle:** true treatment probability from the simulated data-generating mechanism.

## Experiment 1 — clinical-only

Treatment and outcome depend on a clinical-state score formed from SBP, DBP, BMI, and creatinine. The true treatment effect is -5.

This functions as a positive control: when relevant measured clinical confounders are directly available, conventional adjustment should perform near the Oracle.

## Experiment 2 — temporal-only

The 365-day history is summarized using trends and accelerations across six event families, yielding 12 temporal truth variables. Treatment and outcome depend on nonlinear functions of those longitudinal variables. The true treatment effect is -5.

Explicit Temporal is a key benchmark because it tests whether the temporal confounding is recoverable when the relevant temporal information is supplied directly.

## Experiment 3 — mixed clinical + temporal

Clinical state and temporal trajectory both contribute to treatment and outcome. The experiment compares explicit structured adjustment, CLMBR, naive concatenation, and a block-preserving offset hybrid.

The complete current treatment-model comparison includes both AUC and log loss for every propensity-based method.

## Real treatment assignment

The real-data section uses a Metformin-vs-sulfonylurea active-comparator cohort (160 Metformin, 76 sulfonylurea). It compares 19 curated variables, 31 engineered temporal variables, the 768-dimensional CLMBR representation, and a Curated + CLMBR offset model.

Because the true causal effect is unknown, the real-data section evaluates treatment-selection prediction and observed balance, not known-truth causal-effect recovery.

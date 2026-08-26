#!/usr/bin/env python3
"""Reproducible confounding-only EHR foundation-model analysis.

This script implements the confounding-focused analyses in a single reproducible
workflow:

1. Rebuild the common EHRSHOT pseudo-index cohort from raw MEDS events.
2. Rebuild common Curated, Count, clinical, temporal, and CLMBR representations.
3. Experiment 1: clinical-only plasmode positive control.
4. Experiment 2: temporal-only plasmode stress test with Explicit Temporal benchmark.
5. Experiment 3: mixed clinical + temporal plasmode with structured and FM hybrids.
6. Blinded active-comparator feasibility screen and temporal-signal selection.
7. Metformin vs sulfonylurea real cross-fitted treatment model.
8. Observed clinical + temporal balance diagnostics.

Intentionally excluded:
- the full post-treatment indication-expansion / 24-outcome screen,
- AKI / anemia / dyspnea outcome highlights,
- scGPT / single-cell analyses,
- genomic-distance analyses.

Reproducibility principle
-------------------------
All analysis-derived quantities are recomputed at runtime from:

  A) the raw EHRSHOT MEDS parquet dataset, and
  B) the published pretrained CLMBR checkpoint.

Generated result tables are written to --output-dir.

The first-stage comparator screen still counts how many incident condition
endpoints would be available in each pair, because endpoint availability was a
pre-treatment/blinded FEASIBILITY criterion used to select the confounding
comparison. It does NOT estimate any post-treatment outcome effect.

Recommended environment (matching the reference analysis environment as closely as
possible):
    femr==0.2.3
    datasets==2.15.0
    transformers==4.35.2
    numpy==1.26.4
    pandas, pyarrow, scipy, scikit-learn, torch, requests

Examples
--------
Colab after Google Drive is already mounted::

    %run ehr_foundation_model_confounding_reproducible.py \\
        --ehr-root /content/drive/MyDrive/EHRSHOT

Local / server::

    python ehr_foundation_model_confounding_reproducible.py \\
        --ehr-root /path/to/EHRSHOT \\
        --clmbr-model-path /path/to/clmbr-t-base \\
        --output-dir ./confounding_outputs

If the CLMBR repository is not already present, add
--download-clmbr-if-missing. The StanfordShahLab/clmbr-t-base repository may
require Hugging Face authentication/access.

Revision: 2026-08-26
This version includes all three plasmode experiments, complete mixed-model
treatment metrics, fixed-scale SMD diagnostics, non-fatal reference checks,
and a wider-grid naive-hybrid sensitivity analysis.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import gc
import json
import math
import time
import warnings

REPRODUCIBILITY_REVISION = '2026-08-26'

import numpy as np
import pandas as pd

import pyarrow.dataset as ds
import requests

from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reproduce the confounding-only EHR foundation-model analyses from raw EHRSHOT MEDS.'
    )
    parser.add_argument(
        '--ehr-root',
        default='/content/drive/MyDrive/EHRSHOT',
        help='Directory containing EHRSHOT_MEDS/ and, by default, clmbr-t-base/.',
    )
    parser.add_argument(
        '--clmbr-model-path',
        default=None,
        help='Path to pretrained CLMBR model. Default: <ehr-root>/clmbr-t-base',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Where generated result tables are written. Default: <ehr-root>/confounding_reproducible_outputs',
    )
    parser.add_argument(
        '--mount-drive',
        action='store_true',
        help='Mount Google Drive at /content/drive before accessing paths (Colab only).',
    )
    parser.add_argument(
        '--force-remount-drive',
        action='store_true',
        help='Pass force_remount=True to google.colab.drive.mount().',
    )
    parser.add_argument(
        '--download-clmbr-if-missing',
        action='store_true',
        help='Download StanfordShahLab/clmbr-t-base to --clmbr-model-path if missing.',
    )
    return parser.parse_args()


ARGS = parse_args()

if ARGS.mount_drive:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError('--mount-drive was requested, but google.colab is not available.') from exc
    drive.mount('/content/drive', force_remount=ARGS.force_remount_drive)

EHR_ROOT = Path(ARGS.ehr_root).expanduser().resolve()
MEDS_ROOT = EHR_ROOT / 'EHRSHOT_MEDS'
CLMBR_MODEL_PATH = (
    Path(ARGS.clmbr_model_path).expanduser().resolve()
    if ARGS.clmbr_model_path
    else EHR_ROOT / 'clmbr-t-base'
)
OUTPUT_DIR = (
    Path(ARGS.output_dir).expanduser().resolve()
    if ARGS.output_dir
    else EHR_ROOT / 'confounding_reproducible_outputs'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if ARGS.download_clmbr_if_missing and not CLMBR_MODEL_PATH.exists():
    from huggingface_hub import snapshot_download
    print('Downloading StanfordShahLab/clmbr-t-base ...')
    snapshot_download(
        repo_id='StanfordShahLab/clmbr-t-base',
        local_dir=str(CLMBR_MODEL_PATH),
    )

if not MEDS_ROOT.exists():
    raise FileNotFoundError(
        f'EHRSHOT MEDS root not found: {MEDS_ROOT}\n'
        'Supply --ehr-root pointing to a directory that contains EHRSHOT_MEDS/.'
    )
if not CLMBR_MODEL_PATH.exists():
    raise FileNotFoundError(
        f'CLMBR model not found: {CLMBR_MODEL_PATH}\n'
        'Supply --clmbr-model-path or use --download-clmbr-if-missing.'
    )


def show(df, decimals=4):
    """Compact table printing for terminal or notebook execution."""
    if isinstance(df, pd.DataFrame):
        print(df.round(decimals).to_string(index=False))
    else:
        print(df)


def save_table(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print('Saved:', path)
    return path

print('EHR root:', EHR_ROOT)
print('MEDS root:', MEDS_ROOT)
print('CLMBR model:', CLMBR_MODEL_PATH)
print('Output directory:', OUTPUT_DIR)

# ============================================================
# 0A. RAW EHRSHOT DATASET
# ============================================================

DATA_FILES = sorted((MEDS_ROOT / 'data').rglob('*.parquet'))
if not DATA_FILES:
    raise FileNotFoundError(f'No MEDS parquet files found under {MEDS_ROOT / "data"}')

ehr_ds = ds.dataset([str(p) for p in DATA_FILES], format='parquet')
print('Event parquet files:', len(DATA_FILES))
print('Schema:')
print(ehr_ds.schema)

# ============================================================
# 0B. SHARED CAUSAL / MODELING UTILITIES
# ============================================================

def ess(w):
    w = np.asarray(w, dtype=float)
    return (w.sum() ** 2) / np.sum(w ** 2)


def overlap_weights(A, ps):
    A = np.asarray(A, dtype=int)
    ps = np.asarray(ps, dtype=float)
    return np.where(A == 1, 1.0 - ps, ps)


def weighted_effect(y, A, w):
    y = np.asarray(y, dtype=float)
    A = np.asarray(A, dtype=int)
    w = np.asarray(w, dtype=float)
    return (
        np.average(y[A == 1], weights=w[A == 1])
        - np.average(y[A == 0], weights=w[A == 0])
    )


def weighted_smd(x, A, w=None):
    """Standardized mean difference on a fixed pre-weighting scale.

    Weighting changes the treated/control means in the numerator, but the
    denominator is the unweighted pooled baseline SD.  This keeps SMDs
    comparable before versus after weighting and across weighting methods.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=int)
    if w is None:
        w = np.ones(len(A), dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    treated = A == 1
    control = A == 0
    x1, x0 = x[treated], x[control]
    w1, w0 = w[treated], w[control]

    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)

    # Fixed reference scale: unweighted pre-adjustment pooled SD.
    v1_ref = np.var(x1, ddof=1) if len(x1) > 1 else 0.0
    v0_ref = np.var(x0, ddof=1) if len(x0) > 1 else 0.0
    pooled_ref = np.sqrt((v1_ref + v0_ref) / 2.0)
    return 0.0 if pooled_ref == 0 else (m1 - m0) / pooled_ref


def make_logistic(C, max_iter=3000):
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('logit', LogisticRegression(
            penalty='l2', C=C, solver='liblinear', max_iter=max_iter
        )),
    ])


def nested_oof_logistic(X, y, outer_seed, C_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
    """Nested 5x3 cross-fitted logistic regression selected by inner log loss."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    ps = np.full(len(y), np.nan)
    selected_C = []

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=outer_seed + 100 + fold)
        losses = {C: [] for C in C_grid}
        for itr, iva in inner.split(X[tr], y[tr]):
            for C in C_grid:
                m = make_logistic(C)
                m.fit(X[tr][itr], y[tr][itr])
                pred = m.predict_proba(X[tr][iva])[:, 1]
                losses[C].append(log_loss(y[tr][iva], pred, labels=[0, 1]))
        best_C = min(C_grid, key=lambda C: np.mean(losses[C]))
        selected_C.append(best_C)
        final = make_logistic(best_C)
        final.fit(X[tr], y[tr])
        ps[te] = final.predict_proba(X[te])[:, 1]

    assert np.isfinite(ps).all()
    return {
        'ps': ps,
        'auc': roc_auc_score(y, ps),
        'logloss': log_loss(y, ps, labels=[0, 1]),
        'selected_C': selected_C,
    }


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_offset_ridge(X, y, offset, lam, init=None):
    """Logistic correction with FIXED offset; only FM coefficients are L2-penalized."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    n, p = X.shape
    if init is None:
        init = np.zeros(p + 1, dtype=float)

    def objective(theta):
        b, beta = theta[0], theta[1:]
        eta = offset + b + X @ beta
        nll = np.mean(np.logaddexp(0, eta) - y * eta)
        penalty = 0.5 * lam * np.dot(beta, beta)
        p_hat = expit(eta)
        residual = p_hat - y
        grad_b = residual.mean()
        grad_beta = (X.T @ residual) / n + lam * beta
        return nll + penalty, np.concatenate([[grad_b], grad_beta])

    result = minimize(
        fun=lambda th: objective(th)[0],
        x0=init,
        jac=lambda th: objective(th)[1],
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10},
    )
    if not result.success:
        warnings.warn(f'Offset optimizer: {result.message}')
    return result.x


def predict_offset_model(X, offset, theta):
    return expit(np.asarray(offset) + theta[0] + np.asarray(X) @ theta[1:])



def model_logit(model, X):
    """Return the model's linear predictor when available."""
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X)).reshape(-1)
    return safe_logit(model.predict_proba(X)[:, 1])


def select_C(X, y, C_grid, seed):
    """Select C by 3-fold inner cross-validation using log loss."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    losses = {C: [] for C in C_grid}
    for tr, va in inner.split(X, y):
        for C in C_grid:
            m = make_logistic(C)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            losses[C].append(log_loss(y[va], p, labels=[0, 1]))
    return min(C_grid, key=lambda C: np.mean(losses[C]))

print('\n=== 1. REBUILD CELL 3 COHORT ===')

# ============================================================
# 1A. REBUILD THE FROZEN 1,922-PERSON CELL 3 COHORT
# ============================================================

history = ehr_ds.to_table(columns=['subject_id', 'time', 'omop_table']).to_pandas()
history['time'] = pd.to_datetime(history['time'])
all_clinical = history.loc[history['omop_table'] != 'person'].copy()

history_span = (
    all_clinical.groupby('subject_id')['time']
    .agg(['min', 'max', 'count'])
    .reset_index()
)
history_span['clinical_span_days'] = (
    (history_span['max'] - history_span['min']).dt.total_seconds() / 86400.0
)
eligible_cell3 = history_span.loc[history_span['clinical_span_days'] >= 365].copy()

cell3_index = eligible_cell3[['subject_id', 'min', 'max', 'clinical_span_days']].rename(
    columns={'min': 'first_clinical_time', 'max': 'last_clinical_time'}
)
cell3_index['index_date'] = pd.to_datetime(cell3_index['last_clinical_time']).dt.normalize()

cell3_baseline_events = all_clinical.merge(
    cell3_index[['subject_id', 'index_date']], on='subject_id', how='inner'
)
cell3_baseline_events = cell3_baseline_events.loc[
    (cell3_baseline_events['time'] < cell3_baseline_events['index_date'])
    & (cell3_baseline_events['time'] >= cell3_baseline_events['index_date'] - pd.Timedelta(days=365))
].copy()

support = (
    cell3_baseline_events.groupby('subject_id')
    .agg(
        n_events_365=('time', 'size'),
        n_event_dates_365=('time', lambda x: x.dt.normalize().nunique()),
    )
    .reset_index()
)
cell3_index = cell3_index.merge(support, on='subject_id', how='left')
cell3_index[['n_events_365', 'n_event_dates_365']] = (
    cell3_index[['n_events_365', 'n_event_dates_365']].fillna(0)
)

cell3_cohort = (
    cell3_index.loc[cell3_index['n_event_dates_365'] >= 20]
    .sort_values('subject_id')
    .reset_index(drop=True)
)
cell3_ids = set(cell3_cohort['subject_id'])
cell3_events = cell3_baseline_events.loc[
    cell3_baseline_events['subject_id'].isin(cell3_ids)
].copy()

cell3_events['days_pre'] = (
    (cell3_events['index_date'] - cell3_events['time']).dt.total_seconds() / 86400.0
)
cell3_events['time_bin'] = pd.cut(
    cell3_events['days_pre'],
    bins=[0, 90, 180, 270, 365.000001],
    labels=['Q1_recent', 'Q2', 'Q3', 'Q4_old'],
    include_lowest=False,
    right=True,
)

print('Patients with >=365d clinical span:', len(eligible_cell3))
print('Frozen Cell 3 cohort:', len(cell3_cohort))
print('Distinct-date threshold check:', cell3_cohort['n_event_dates_365'].min())
print(cell3_events['time_bin'].value_counts(sort=False))

if len(cell3_cohort) != 1922:
    warnings.warn(
        f'Reference run used 1,922 Cell-3 patients; this run produced '
        f'{len(cell3_cohort):,}. Check the EHRSHOT release and preprocessing '
        'if exact numerical reproduction is required.'
    )

# ============================================================
# 1B. TEMPORAL TRUTH FEATURES: TRENDS + ACCELERATIONS
# ============================================================

TIME_BINS = ['Q1_recent', 'Q2', 'Q3', 'Q4_old']
TABLE_TYPES = {
    'condition': 'condition_occurrence',
    'drug': 'drug_exposure',
    'measurement': 'measurement',
    'procedure': 'procedure_occurrence',
}
PREFIXES = ['all_events', 'event_dates', 'condition', 'drug', 'measurement', 'procedure']

cell3_temporal = pd.DataFrame(index=cell3_cohort['subject_id'].astype(int))
cell3_temporal.index.name = 'subject_id'

# All-event counts by temporal quarter.
counts = pd.crosstab(cell3_events['subject_id'], cell3_events['time_bin'])
counts = counts.reindex(columns=TIME_BINS, fill_value=0)
counts.columns = [f'all_events_{b}' for b in TIME_BINS]
cell3_temporal = cell3_temporal.join(counts)

# Distinct clinical dates by quarter.
tmp = cell3_events.copy()
tmp['event_date'] = pd.to_datetime(tmp['time']).dt.normalize()
dates = (
    tmp.groupby(['subject_id', 'time_bin'], observed=True)['event_date']
    .nunique().unstack(fill_value=0).reindex(columns=TIME_BINS, fill_value=0)
)
dates.columns = [f'event_dates_{b}' for b in TIME_BINS]
cell3_temporal = cell3_temporal.join(dates)

# Major event-type counts by quarter.
for short, omop in TABLE_TYPES.items():
    x = cell3_events.loc[cell3_events['omop_table'] == omop]
    c = pd.crosstab(x['subject_id'], x['time_bin']).reindex(columns=TIME_BINS, fill_value=0)
    c.columns = [f'{short}_{b}' for b in TIME_BINS]
    cell3_temporal = cell3_temporal.join(c)

cell3_temporal = cell3_temporal.fillna(0)
temporal_truth_cols = []
for prefix in PREFIXES:
    q1 = np.log1p(cell3_temporal[f'{prefix}_Q1_recent'])
    q2 = np.log1p(cell3_temporal[f'{prefix}_Q2'])
    q3 = np.log1p(cell3_temporal[f'{prefix}_Q3'])
    q4 = np.log1p(cell3_temporal[f'{prefix}_Q4_old'])
    trend = f'{prefix}_trend'
    accel = f'{prefix}_acceleration'
    cell3_temporal[trend] = q1 - q4
    cell3_temporal[accel] = (q1 - q2) - (q3 - q4)
    temporal_truth_cols += [trend, accel]

print('Temporal table:', cell3_temporal.shape)
print('Truth variables:', temporal_truth_cols)
assert len(temporal_truth_cols) == 12
assert np.isfinite(cell3_temporal.to_numpy()).all()

# ============================================================
# 1C. COMMON CURATED + COUNT REPRESENTATIONS
# ============================================================

# Curated annual aggregates: the six quarter-count families collapsed across time.
curated_rows = pd.DataFrame(index=cell3_temporal.index)
for prefix in PREFIXES:
    total = sum(cell3_temporal[f'{prefix}_{b}'] for b in TIME_BINS)
    curated_rows[f'log_{prefix}_365'] = np.log1p(total)
cell3_curated = curated_rows.reset_index()
X3_curated = cell3_curated.drop(columns='subject_id').to_numpy(dtype=float)

# Sparse count representation: exact baseline OMOP table::code counts, log1p transformed.
count_source = cell3_events[['subject_id', 'omop_table', 'code']].copy()
count_source['feature'] = count_source['omop_table'].astype(str) + '::' + count_source['code'].astype(str)
patient_dicts = []
for sid in cell3_cohort['subject_id']:
    vc = count_source.loc[count_source['subject_id'] == sid, 'feature'].value_counts()
    patient_dicts.append(vc.to_dict())

count_vectorizer = DictVectorizer(sparse=True)
X3_count = count_vectorizer.fit_transform(patient_dicts).tocsr().astype(np.float64)
X3_count.data = np.log1p(X3_count.data)

print('Curated:', X3_curated.shape)
print('Count:', X3_count.shape)

# ============================================================
# 1D. GENERIC PRE-INDEX CLMBR EXTRACTION HELPERS
#     ALWAYS COMPUTED FRESH -- NO DISK CACHE IS READ
# ============================================================

_CLMBR_RUNTIME = None


def move_to_device(x, device):
    import torch
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def load_clmbr_model():
    """Load the published pretrained CLMBR checkpoint; cache only in RAM for this process."""
    global _CLMBR_RUNTIME
    if _CLMBR_RUNTIME is not None:
        return _CLMBR_RUNTIME

    import torch
    import femr.models.transformer
    import femr.models.tokenizer
    import femr.models.processor

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(str(CLMBR_MODEL_PATH))
    batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)
    model = femr.models.transformer.FEMRModel.from_pretrained(str(CLMBR_MODEL_PATH))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    _CLMBR_RUNTIME = (model, tokenizer, batch_processor, device)
    print('CLMBR inference device:', device)
    return _CLMBR_RUNTIME


def prepare_clmbr_source(cohort, index_col, window_days=365):
    ids = cohort['subject_id'].astype(int).tolist()
    raw = ehr_ds.to_table(
        columns=['subject_id', 'time', 'code', 'numeric_value', 'text_value', 'unit', 'omop_table'],
        filter=ds.field('subject_id').isin(ids),
    ).to_pandas()
    raw['time'] = pd.to_datetime(raw['time'])
    idx = cohort[['subject_id', index_col]].copy().rename(columns={index_col: 'index_time'})
    idx['index_time'] = pd.to_datetime(idx['index_time'])
    raw = raw.merge(idx, on='subject_id', how='inner', validate='many_to_one')

    is_person = raw['omop_table'].eq('person')
    in_window = (
        ~is_person
        & (raw['time'] < raw['index_time'])
        & (raw['time'] >= raw['index_time'] - pd.Timedelta(days=window_days))
    )
    source = raw.loc[is_person | in_window].copy()

    # EHRSHOT -> FEMR birth-anchor compatibility transform required for model input.
    source.loc[source['code'].eq('SNOMED/3950001'), 'code'] = 'SNOMED/184099003'
    source = source.sort_values(['subject_id', 'time']).reset_index(drop=True)
    return source


def build_femr_patient(g, sid):
    events = []
    for event_time, rows in g.groupby('time', sort=True):
        measurements = []
        for row in rows.itertuples(index=False):
            m = {'code': str(row.code)}
            if pd.notna(row.numeric_value):
                m['numeric_value'] = float(row.numeric_value)
            if pd.notna(row.text_value):
                m['text_value'] = str(row.text_value)
            if hasattr(row, 'unit') and pd.notna(row.unit):
                m['unit'] = str(row.unit)
            measurements.append(m)
        events.append({
            'time': pd.Timestamp(event_time).to_pydatetime(),
            'measurements': measurements,
        })
    return {'patient_id': int(sid), 'events': events}


def extract_clmbr_matrix(cohort, index_col, window_days=365):
    """Compute one 768-d pre-index CLMBR vector per patient from raw EHRSHOT events."""
    import torch

    model, tokenizer, batch_processor, device = load_clmbr_model()
    source = prepare_clmbr_source(cohort, index_col, window_days=window_days)
    groups = {int(sid): g for sid, g in source.groupby('subject_id', sort=False)}

    embeddings, meta_rows = [], []
    for i, row in enumerate(cohort[['subject_id', index_col]].itertuples(index=False), start=1):
        sid = int(row[0])
        index_time = pd.Timestamp(row[1])
        if sid not in groups:
            raise RuntimeError(f'No pre-index source rows were found for subject {sid}.')
        g = groups[sid].sort_values('time')
        patient = build_femr_patient(g, sid)
        raw_batch = batch_processor.convert_patient(patient, tensor_type='pt')
        batch = move_to_device(batch_processor.collate([raw_batch]), device)
        with torch.no_grad():
            _, result = model(**batch)

        reps = result['representations']
        if torch.is_tensor(reps):
            reps = reps.detach().cpu().numpy()
        reps = np.asarray(reps)
        if reps.ndim == 3 and reps.shape[0] == 1:
            reps = reps[0]
        if reps.ndim != 2:
            raise ValueError(f'Unexpected representation shape for {sid}: {reps.shape}')

        # The source history is strictly pre-index, so the final representation
        # position is the last available pre-index state.
        emb = reps[-1].astype(np.float32)
        if emb.shape != (768,):
            raise ValueError(f'Unexpected embedding shape for {sid}: {emb.shape}')

        latest_source_time = pd.to_datetime(
            g.loc[g['omop_table'] != 'person', 'time']
        ).max()
        gap = (index_time - latest_source_time).total_seconds() / 86400.0
        embeddings.append(emb)
        meta_rows.append({
            'subject_id': sid,
            'latest_source_time': latest_source_time,
            'representation_gap_days': gap,
            'n_raw_rows': len(g),
            'n_rep_positions': reps.shape[0],
        })

        if i % 50 == 0 or i == len(cohort):
            print(f'CLMBR: {i}/{len(cohort)}')

        del patient, raw_batch, batch, result, reps, emb
        if torch.cuda.is_available() and i % 25 == 0:
            torch.cuda.empty_cache()
        if i % 25 == 0:
            gc.collect()

    X = np.vstack(embeddings)
    meta = pd.DataFrame(meta_rows)
    return X, meta

# ============================================================
# 1E. COMMON CLMBR REPRESENTATION + INDEX-ANCHOR FEATURE
# ============================================================

X3_clmbr, meta3 = extract_clmbr_matrix(
    cell3_cohort[['subject_id', 'index_date']],
    'index_date',
    window_days=365,
)

meta3 = meta3.set_index('subject_id').loc[cell3_cohort['subject_id']].reset_index()
rep_gap3 = pd.to_numeric(meta3['representation_gap_days'], errors='coerce').fillna(365).to_numpy()
X3_clmbr_anchored = np.column_stack([
    X3_clmbr,
    np.log1p(np.maximum(rep_gap3, 0)),
])

print('CLMBR:', X3_clmbr.shape)
print('CLMBR anchored:', X3_clmbr_anchored.shape)
if X3_clmbr.shape != (len(cell3_cohort), 768):
    raise RuntimeError(
        f'Expected CLMBR matrix {(len(cell3_cohort), 768)}, found {X3_clmbr.shape}.'
    )
if X3_clmbr_anchored.shape != (len(cell3_cohort), 769):
    raise RuntimeError(
        f'Expected anchored CLMBR matrix {(len(cell3_cohort), 769)}, '
        f'found {X3_clmbr_anchored.shape}.'
    )

# ============================================================
# 1F. COMMON PROPENSITY-MODEL SPECIFICATIONS
# ============================================================

outer_cv3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

curated3_model = Pipeline([
    ('scale', StandardScaler()),
    ('logit', LogisticRegressionCV(
        Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        cv=3, penalty='l2', solver='lbfgs', scoring='neg_log_loss',
        max_iter=5000, n_jobs=-1,
    )),
])
count3_model = Pipeline([
    ('scale', StandardScaler(with_mean=False)),
    ('logit', LogisticRegressionCV(
        Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        cv=3, penalty='l1', solver='liblinear', scoring='neg_log_loss',
        max_iter=5000, n_jobs=-1,
    )),
])
clmbr3_model = Pipeline([
    ('scale', StandardScaler()),
    ('logit', LogisticRegressionCV(
        Cs=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        cv=3, penalty='l2', solver='lbfgs', scoring='neg_log_loss',
        max_iter=5000, n_jobs=-1,
    )),
])


# ============================================================
# 1G. BUILD THE EXPLICIT CLINICAL BLOCK FROM RAW EHR
# ============================================================
# This block is shared by the clinical-only and mixed-confounding experiments.
# Truth generation uses the latest valid pre-index SBP, DBP, BMI, and creatinine.
# Missingness indicators are available to the fitted Curated model, but
# missingness itself does not enter the truth-generating clinical score.

clmbr_source3 = prepare_clmbr_source(
    cell3_cohort[['subject_id', 'index_date']],
    'index_date',
    365,
)
clinical3b_source = clmbr_source3.loc[
    clmbr_source3['omop_table'] != 'person'
].copy()
clinical3b_source['days_pre'] = (
    (clinical3b_source['index_time'] - clinical3b_source['time'])
    .dt.total_seconds() / 86400.0
)

CLINICAL_TRUTH = {
    'SBP': ('LOINC/8480-6', 50, 300),
    'DBP': ('LOINC/8462-4', 30, 200),
    'BMI': ('LOINC/39156-5', 10, 80),
    'Creatinine': ('LOINC/2160-0', 0.1, 20),
}

clinical_latest = pd.DataFrame({
    'subject_id': cell3_cohort['subject_id'].astype(int).to_numpy()
})

for name, (loinc, lower, upper) in CLINICAL_TRUTH.items():
    g = clinical3b_source.loc[
        clinical3b_source['code'].eq(loinc),
        ['subject_id', 'days_pre', 'numeric_value'],
    ].copy()
    g['numeric_value'] = pd.to_numeric(g['numeric_value'], errors='coerce')
    g = g.loc[g['numeric_value'].between(lower, upper)]

    if len(g):
        nearest = (
            g.sort_values('days_pre')
            .drop_duplicates('subject_id', keep='first')
            .rename(columns={
                'numeric_value': f'{name}_baseline',
                'days_pre': f'{name}_recency_days',
            })
            [['subject_id', f'{name}_baseline', f'{name}_recency_days']]
        )
    else:
        nearest = pd.DataFrame(columns=[
            'subject_id', f'{name}_baseline', f'{name}_recency_days'
        ])

    clinical_latest = clinical_latest.merge(
        nearest,
        on='subject_id',
        how='left',
        validate='one_to_one',
    )

clinical_value_cols = [f'{x}_baseline' for x in CLINICAL_TRUTH]
for col in clinical_value_cols:
    clinical_latest[f'{col}_missing'] = clinical_latest[col].isna().astype(int)


def truth_z_clinical(series):
    """Winsorize observed values, median-impute, then standardize."""
    x = pd.to_numeric(series, errors='coerce').astype(float)
    observed = x.dropna()
    if observed.empty:
        return np.zeros(len(x), dtype=float)
    lo = observed.quantile(0.01)
    hi = observed.quantile(0.99)
    x = x.clip(lower=lo, upper=hi)
    x = x.fillna(x.median())
    sd = x.std()
    if pd.isna(sd) or sd == 0:
        return np.zeros(len(x), dtype=float)
    return ((x - x.mean()) / sd).to_numpy()


z_sbp = truth_z_clinical(clinical_latest['SBP_baseline'])
z_dbp = truth_z_clinical(clinical_latest['DBP_baseline'])
z_bmi = truth_z_clinical(clinical_latest['BMI_baseline'])
z_creatinine = truth_z_clinical(clinical_latest['Creatinine_baseline'])

clinical_score_raw = (z_sbp + z_dbp + z_bmi + z_creatinine) / 4.0
clinical_score = (
    clinical_score_raw - clinical_score_raw.mean()
) / clinical_score_raw.std()

clinical_truth = pd.DataFrame({
    'z_SBP': z_sbp,
    'z_DBP': z_dbp,
    'z_BMI': z_bmi,
    'z_Creatinine': z_creatinine,
})

clinical_model_cols = (
    [f'{x}_baseline' for x in CLINICAL_TRUTH]
    + [f'{x}_recency_days' for x in CLINICAL_TRUTH]
    + [f'{x}_baseline_missing' for x in CLINICAL_TRUTH]
)
clinical_model = clinical_latest[clinical_model_cols].copy()
for col in [c for c in clinical_model_cols if not c.endswith('_missing')]:
    clinical_model[col] = clinical_model[col].fillna(clinical_model[col].median())

X_clinical_explicit = clinical_model.to_numpy(dtype=float)
X_curated_clinical = np.hstack([X3_curated, X_clinical_explicit])

print('Explicit clinical block:', X_clinical_explicit.shape)
print('Curated clinical representation:', X_curated_clinical.shape)
if X_curated_clinical.shape[1] != 18:
    raise RuntimeError(
        f'Expected 18 Curated clinical features, found {X_curated_clinical.shape[1]}.'
    )

print('\n=== EXPERIMENT 1. CLINICAL-ONLY CONFOUNDING POSITIVE CONTROL ===')

# ============================================================
# 2A. CLINICAL-ONLY TREATMENT + OUTCOME DGP
# ============================================================

TARGET_PREVALENCE_CLIN = 0.40
TARGET_PS_TAIL_RATE_CLIN = 0.05
TARGET_CRUDE_BIAS_CLIN = 3.0
TRUE_TAU_CLIN = -5.0
TREATMENT_SEED_CLIN = 20260818

C_CLIN = np.asarray(clinical_score, dtype=float)


def solve_intercept_clin(gamma):
    lp = gamma * C_CLIN
    low, high = -15.0, 15.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if expit(mid + lp).mean() > TARGET_PREVALENCE_CLIN:
            high = mid
        else:
            low = mid
    intercept = (low + high) / 2.0
    return intercept, expit(intercept + lp)


gamma_low, gamma_high = 0.01, 5.0
for _ in range(100):
    gamma_mid = (gamma_low + gamma_high) / 2.0
    _, p_mid = solve_intercept_clin(gamma_mid)
    tail_rate = np.mean((p_mid < 0.05) | (p_mid > 0.95))
    if tail_rate > TARGET_PS_TAIL_RATE_CLIN:
        gamma_high = gamma_mid
    else:
        gamma_low = gamma_mid

GAMMA_CLIN = (gamma_low + gamma_high) / 2.0
INTERCEPT_CLIN, ps_oracle_clin = solve_intercept_clin(GAMMA_CLIN)

rng_treatment_clin = np.random.default_rng(TREATMENT_SEED_CLIN)
A_clin = rng_treatment_clin.binomial(1, ps_oracle_clin).astype(int)

delta_C_clin = (
    C_CLIN[A_clin == 1].mean()
    - C_CLIN[A_clin == 0].mean()
)
BETA_CLIN = TARGET_CRUDE_BIAS_CLIN / delta_C_clin
mu0_clin = 10.0 + BETA_CLIN * C_CLIN
y_expected_clin = mu0_clin + TRUE_TAU_CLIN * A_clin

# ============================================================
# 2B. CLINICAL-ONLY PROPENSITY MODELS
# ============================================================

ps_curated_clin = cross_val_predict(
    clone(curated3_model),
    X_curated_clinical,
    A_clin,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps_count_clin = cross_val_predict(
    clone(count3_model),
    X3_count,
    A_clin,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps_clmbr_clin = cross_val_predict(
    clone(clmbr3_model),
    X3_clmbr_anchored,
    A_clin,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]

ow_curated_clin = overlap_weights(A_clin, ps_curated_clin)
ow_count_clin = overlap_weights(A_clin, ps_count_clin)
ow_clmbr_clin = overlap_weights(A_clin, ps_clmbr_clin)
ow_oracle_clin = overlap_weights(A_clin, ps_oracle_clin)

methods_clin = {
    'Oracle': (ps_oracle_clin, ow_oracle_clin),
    'Curated': (ps_curated_clin, ow_curated_clin),
    'Count': (ps_count_clin, ow_count_clin),
    'CLMBR': (ps_clmbr_clin, ow_clmbr_clin),
    'Crude': (None, np.ones(len(A_clin))),
}

truth_matrix_clin = clinical_truth.to_numpy(dtype=float)
crude_expected_clin = weighted_effect(
    y_expected_clin, A_clin, np.ones(len(A_clin))
)
crude_bias_clin = abs(crude_expected_clin - TRUE_TAU_CLIN)

clinical_rows = []
clinical_balance_rows = []
for method, (ps, w) in methods_clin.items():
    estimate = weighted_effect(y_expected_clin, A_clin, w)
    signed_bias = estimate - TRUE_TAU_CLIN
    smds = np.abs([
        weighted_smd(truth_matrix_clin[:, j], A_clin, w)
        for j in range(truth_matrix_clin.shape[1])
    ])
    clinical_rows.append({
        'Method': method,
        'OOF_AUC': np.nan if ps is None else roc_auc_score(A_clin, ps),
        'OOF_logloss': np.nan if ps is None else log_loss(A_clin, ps, labels=[0, 1]),
        'Estimate_no_noise': estimate,
        'Bias_no_noise': signed_bias,
        'Bias_reduction_pct': (
            100.0 * (crude_bias_clin - abs(signed_bias)) / crude_bias_clin
        ),
        'Mean_abs_truth_SMD': smds.mean(),
        'Max_abs_truth_SMD': smds.max(),
        'Truth_SMD_gt_0.10': int((smds > 0.10).sum()),
    })
    for j, variable in enumerate(clinical_truth.columns):
        clinical_balance_rows.append({
            'Method': method,
            'Variable': variable,
            'SMD_after': weighted_smd(
                truth_matrix_clin[:, j], A_clin, w
            ),
        })

clinical_only_no_noise = pd.DataFrame(clinical_rows)
clinical_only_balance = pd.DataFrame(clinical_balance_rows)

print('Treatment prevalence:', round(A_clin.mean(), 3))
show(clinical_only_no_noise, 4)

save_table(clinical_only_no_noise, 'clinical_only_no_noise_summary.csv')
save_table(clinical_only_balance, 'clinical_only_truth_balance.csv')

# ============================================================
# 2C. CLINICAL-ONLY 500 OUTCOME SIMULATIONS
# ============================================================

N_SIM_CLIN = 500
NOISE_SD_CLIN = 2.0
OUTCOME_SEED_CLIN = 20260820
rng_outcome_clin = np.random.default_rng(OUTCOME_SEED_CLIN)

clinical_sim_rows = []
for sim_id in range(N_SIM_CLIN):
    epsilon = rng_outcome_clin.normal(
        0.0, NOISE_SD_CLIN, size=len(A_clin)
    )
    y = mu0_clin + TRUE_TAU_CLIN * A_clin + epsilon
    for method, (_, w) in methods_clin.items():
        clinical_sim_rows.append({
            'simulation': sim_id,
            'method': method,
            'estimate': weighted_effect(y, A_clin, w),
        })

clinical_only_simulations = pd.DataFrame(clinical_sim_rows)
clinical_only_simulations['bias'] = (
    clinical_only_simulations['estimate'] - TRUE_TAU_CLIN
)
clinical_only_simulations['squared_error'] = (
    clinical_only_simulations['bias'] ** 2
)
clinical_only_summary = (
    clinical_only_simulations
    .groupby('method')
    .agg(
        mean_estimate=('estimate', 'mean'),
        mean_bias=('bias', 'mean'),
        MC_SD=('estimate', 'std'),
        RMSE=('squared_error', lambda x: np.sqrt(x.mean())),
    )
    .reset_index()
)
clinical_only_summary['abs_bias'] = clinical_only_summary['mean_bias'].abs()
crude_abs_clin = float(
    clinical_only_summary.loc[
        clinical_only_summary['method'].eq('Crude'), 'abs_bias'
    ].iloc[0]
)
clinical_only_summary['bias_reduction_vs_crude_pct'] = (
    100.0
    * (crude_abs_clin - clinical_only_summary['abs_bias'])
    / crude_abs_clin
)
clinical_only_summary = clinical_only_summary.sort_values(
    'abs_bias'
).reset_index(drop=True)

show(clinical_only_summary, 3)
save_table(clinical_only_simulations, 'clinical_only_500_simulations.csv')
save_table(clinical_only_summary, 'clinical_only_summary.csv')

clinical_reference = {
    'Oracle': -4.928,
    'Curated': -4.885,
    'CLMBR': -2.581,
    'Count': -2.207,
    'Crude': -2.000,
}
for method, target in clinical_reference.items():
    got = float(
        clinical_only_summary.loc[
            clinical_only_summary['method'].eq(method),
            'mean_estimate',
        ].iloc[0]
    )
    print(
        f'{method:8s}: computed={got:+.3f}, '
        f'reference={target:+.3f}, delta={got-target:+.3f}'
    )

print('\n=== EXPERIMENT 2. TEMPORAL-ONLY CONFOUNDING STRESS TEST ===')

# ============================================================
# 3A. STANDARDIZE THE 12 TEMPORAL TRUTH VARIABLES
# ============================================================

from scipy.optimize import brentq

truth3a = cell3_temporal[temporal_truth_cols].copy()
Z3 = StandardScaler().fit_transform(truth3a)
truth_balance3 = pd.DataFrame(
    Z3,
    index=truth3a.index,
    columns=[f'z_{c}' for c in temporal_truth_cols],
)
z = {c: truth_balance3[f'z_{c}'].to_numpy() for c in temporal_truth_cols}

# ============================================================
# 3B. FREEZE TEMPORAL TREATMENT + OUTCOME DGP
# ============================================================

lp_no_intercept3 = (
    0.55 * z['event_dates_trend']
    + 0.65 * z['drug_trend']
    + 0.50 * z['condition_acceleration']
    + 0.55 * z['measurement_acceleration']
    - 0.35 * z['procedure_trend']
    + 0.40 * (z['drug_trend'] * z['event_dates_trend'])
    - 0.35 * (
        z['condition_acceleration'] * z['measurement_acceleration']
    )
    + 0.30 * np.tanh(z['all_events_acceleration'])
    + 0.25 * (z['drug_acceleration'] ** 2 - 1.0)
)

TARGET_PREVALENCE_3A = 0.40
intercept3 = brentq(
    lambda b: expit(b + lp_no_intercept3).mean() - TARGET_PREVALENCE_3A,
    -20,
    20,
)
true_ps3 = expit(intercept3 + lp_no_intercept3)

rng_treat3 = np.random.default_rng(314159)
A3 = rng_treat3.binomial(1, true_ps3)

TRUE_TAU_3A = -5.0
mu0_3a = (
    10.0
    + 0.80 * z['event_dates_trend']
    + 0.90 * z['drug_trend']
    + 0.80 * z['condition_acceleration']
    + 0.80 * z['measurement_acceleration']
    + 0.40 * z['all_events_acceleration']
    + 0.50 * (z['drug_trend'] * z['condition_acceleration'])
    + 0.40 * (
        z['measurement_acceleration'] * z['event_dates_trend']
    )
)
y_expected3 = mu0_3a + TRUE_TAU_3A * A3
oracle_ow3 = overlap_weights(A3, true_ps3)

print('Treatment prevalence:', round(A3.mean(), 3), f'({A3.sum()}/{len(A3)})')
print('Treatment intercept:', round(intercept3, 3))
print(
    'Expected crude estimate:',
    round(weighted_effect(y_expected3, A3, np.ones(len(A3))), 3),
)
print(
    'Expected oracle estimate:',
    round(weighted_effect(y_expected3, A3, oracle_ow3), 3),
)

# ============================================================
# 3C. FIT CURATED, COUNT, CLMBR, AND EXPLICIT TEMPORAL MODELS
# ============================================================

X3_temporal_explicit = truth_balance3.to_numpy(dtype=float)

ps3_curated = cross_val_predict(
    clone(curated3_model),
    X3_curated,
    A3,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3_count = cross_val_predict(
    clone(count3_model),
    X3_count,
    A3,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3_clmbr = cross_val_predict(
    clone(clmbr3_model),
    X3_clmbr_anchored,
    A3,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3_temporal = cross_val_predict(
    clone(curated3_model),
    X3_temporal_explicit,
    A3,
    cv=outer_cv3,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]

ow3_curated = overlap_weights(A3, ps3_curated)
ow3_count = overlap_weights(A3, ps3_count)
ow3_clmbr = overlap_weights(A3, ps3_clmbr)
ow3_temporal = overlap_weights(A3, ps3_temporal)

cell3a_model_metrics = pd.DataFrame([
    {
        'Method': 'Count',
        'OOF_AUC': roc_auc_score(A3, ps3_count),
        'OOF_logloss': log_loss(A3, ps3_count, labels=[0, 1]),
    },
    {
        'Method': 'Curated',
        'OOF_AUC': roc_auc_score(A3, ps3_curated),
        'OOF_logloss': log_loss(A3, ps3_curated, labels=[0, 1]),
    },
    {
        'Method': 'CLMBR',
        'OOF_AUC': roc_auc_score(A3, ps3_clmbr),
        'OOF_logloss': log_loss(A3, ps3_clmbr, labels=[0, 1]),
    },
    {
        'Method': 'Explicit Temporal',
        'OOF_AUC': roc_auc_score(A3, ps3_temporal),
        'OOF_logloss': log_loss(A3, ps3_temporal, labels=[0, 1]),
    },
    {
        'Method': 'Oracle',
        'OOF_AUC': roc_auc_score(A3, true_ps3),
        'OOF_logloss': log_loss(A3, true_ps3, labels=[0, 1]),
    },
])
show(cell3a_model_metrics, 4)
save_table(cell3a_model_metrics, 'cell3A_treatment_model_metrics.csv')

# Fixed-scale truth balance.
weights3a = {
    'Crude': np.ones(len(A3)),
    'Curated': ow3_curated,
    'Count': ow3_count,
    'CLMBR': ow3_clmbr,
    'Explicit Temporal': ow3_temporal,
    'Oracle': oracle_ow3,
}
truth_matrix3a = truth_balance3.to_numpy(dtype=float)
balance3a_rows = []
for method, w in weights3a.items():
    smds = np.abs([
        weighted_smd(truth_matrix3a[:, j], A3, w)
        for j in range(truth_matrix3a.shape[1])
    ])
    balance3a_rows.append({
        'Method': method,
        'Mean_abs_SMD': smds.mean(),
        'Max_abs_SMD': smds.max(),
        'N_SMD_gt_0.10': int((smds > 0.10).sum()),
    })
cell3a_balance = pd.DataFrame(balance3a_rows)
save_table(cell3a_balance, 'cell3A_truth_balance.csv')

# ============================================================
# 3D. 500 OUTCOME SIMULATIONS
# ============================================================

rng_y3 = np.random.default_rng(2027)
sim_rows = []
for sim in range(500):
    eps = rng_y3.normal(0.0, 2.0, size=len(A3))
    Y = mu0_3a + TRUE_TAU_3A * A3 + eps
    for method, w in weights3a.items():
        sim_rows.append({
            'simulation': sim,
            'method': method,
            'estimate': weighted_effect(Y, A3, w),
        })

cell3a_sims = pd.DataFrame(sim_rows)
cell3a_sims['bias'] = cell3a_sims['estimate'] - TRUE_TAU_3A
cell3a_sims['squared_error'] = cell3a_sims['bias'] ** 2
cell3a = (
    cell3a_sims.groupby('method')
    .agg(
        mean_estimate=('estimate', 'mean'),
        mean_bias=('bias', 'mean'),
        MC_SD=('estimate', 'std'),
        RMSE=('squared_error', lambda x: np.sqrt(x.mean())),
    )
    .reset_index()
)
cell3a['abs_bias'] = cell3a['mean_bias'].abs()
crude_abs_bias = float(
    cell3a.loc[cell3a['method'].eq('Crude'), 'abs_bias'].iloc[0]
)
cell3a['bias_reduction_vs_crude_pct'] = (
    100.0 * (crude_abs_bias - cell3a['abs_bias']) / crude_abs_bias
)
cell3a = cell3a.sort_values('abs_bias').reset_index(drop=True)

show(cell3a, 3)
save_table(cell3a_sims, 'cell3A_500_simulations.csv')
save_table(cell3a, 'cell3A_summary.csv')

expected_3a = {
    'Explicit Temporal': -4.974,
    'Oracle': -4.963,
    'CLMBR': -2.651,
    'Curated': -2.470,
    'Count': -2.435,
    'Crude': -2.376,
}
for method, target in expected_3a.items():
    got = float(
        cell3a.loc[cell3a['method'].eq(method), 'mean_estimate'].iloc[0]
    )
    print(
        f'{method:17s}: computed={got:+.3f}, '
        f'reference={target:+.3f}, delta={got-target:+.3f}'
    )

print('\n=== EXPERIMENT 3. MIXED CLINICAL + TEMPORAL CONFOUNDING ===')

# ============================================================
# 4A. MIXED REPRESENTATIONS
# ============================================================

temporal_vars3b = [
    'z_drug_trend',
    'z_event_dates_trend',
    'z_condition_acceleration',
    'z_measurement_acceleration',
]
T_parts = [truth_balance3[c].to_numpy() for c in temporal_vars3b]
temporal_score_raw = sum(T_parts) / 4.0
temporal_score = (
    temporal_score_raw - temporal_score_raw.mean()
) / temporal_score_raw.std()

X3b_curated = X_curated_clinical.copy()
X3b_temporal_explicit = X3_temporal_explicit.copy()
X3b_structured = np.hstack([
    X3b_curated,
    X3b_temporal_explicit,
])
X3b_hybrid = np.hstack([
    X3b_curated,
    X3_clmbr_anchored,
])

print(
    'Clinical-temporal correlation:',
    round(np.corrcoef(clinical_score, temporal_score)[0, 1], 3),
)
print('Curated:', X3b_curated.shape)
print('Explicit Temporal:', X3b_temporal_explicit.shape)
print('Clinical + Temporal:', X3b_structured.shape)
print('CLMBR:', X3_clmbr_anchored.shape)
print('Naive Hybrid:', X3b_hybrid.shape)

if X3b_curated.shape[1] != 18:
    raise RuntimeError('Curated mixed representation must contain 18 features.')
if X3b_temporal_explicit.shape[1] != 12:
    raise RuntimeError('Explicit Temporal representation must contain 12 features.')
if X3b_structured.shape[1] != 30:
    raise RuntimeError('Clinical + Temporal representation must contain 30 features.')
if X3b_hybrid.shape[1] != 787:
    raise RuntimeError('Naive Hybrid representation must contain 787 features.')

# ============================================================
# 4B. FREEZE MIXED TREATMENT + OUTCOME DGP
# ============================================================

C3 = np.asarray(clinical_score, dtype=float)
T3 = np.asarray(temporal_score, dtype=float)
TARGET_PREVALENCE_3B = 0.40
TARGET_PS_TAIL_RATE_3B = 0.05


def solve_intercept_3b(gamma):
    lp = gamma * C3 + gamma * T3
    lo, hi = -15.0, 15.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if expit(mid + lp).mean() > TARGET_PREVALENCE_3B:
            hi = mid
        else:
            lo = mid
    b = (lo + hi) / 2.0
    return b, expit(b + lp)


glo, ghi = 0.01, 3.0
for _ in range(100):
    gmid = (glo + ghi) / 2.0
    _, p = solve_intercept_3b(gmid)
    tail = np.mean((p < 0.05) | (p > 0.95))
    if tail > TARGET_PS_TAIL_RATE_3B:
        ghi = gmid
    else:
        glo = gmid

GAMMA_3B = (glo + ghi) / 2.0
INTERCEPT_3B, ps3b_oracle = solve_intercept_3b(GAMMA_3B)

rng3b = np.random.default_rng(20260818)
A3b = rng3b.binomial(1, ps3b_oracle)

TRUE_TAU_3B = -5.0
TARGET_CRUDE_BIAS_3B = 3.0
delta_C = C3[A3b == 1].mean() - C3[A3b == 0].mean()
delta_T = T3[A3b == 1].mean() - T3[A3b == 0].mean()
BETA_OUTCOME_3B = TARGET_CRUDE_BIAS_3B / (delta_C + delta_T)
mu0_3b = (
    10.0
    + BETA_OUTCOME_3B * C3
    + BETA_OUTCOME_3B * T3
)
y_expected_3b = mu0_3b + TRUE_TAU_3B * A3b
ow3b_oracle = overlap_weights(A3b, ps3b_oracle)

print('gamma:', round(GAMMA_3B, 4))
print('beta:', round(BETA_OUTCOME_3B, 3))
print('Treatment counts:', np.bincount(A3b))
print(
    'Expected crude:',
    round(weighted_effect(y_expected_3b, A3b, np.ones(len(A3b))), 3),
)
print(
    'Expected oracle:',
    round(weighted_effect(y_expected_3b, A3b, ow3b_oracle), 3),
)

# ============================================================
# 4C. CROSS-FITTED PRIMARY REPRESENTATION BENCHMARKS
# ============================================================

cv3b = outer_cv3

ps3b_curated = cross_val_predict(
    clone(curated3_model),
    X3b_curated,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3b_temporal = cross_val_predict(
    clone(curated3_model),
    X3b_temporal_explicit,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3b_structured = cross_val_predict(
    clone(curated3_model),
    X3b_structured,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3b_count = cross_val_predict(
    clone(count3_model),
    X3_count,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ps3b_clmbr = cross_val_predict(
    clone(clmbr3_model),
    X3_clmbr_anchored,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]

# Primary naive-hybrid specification (kept for continuity with the presented analysis):
# use the same strongly regularized high-dimensional C grid as CLMBR alone.
# Because this regularization grid could contribute to the naive-hybrid failure,
# Section 4E immediately re-fits the same representation with a substantially wider
# C grid. Interpret the primary finding alongside that sensitivity analysis.
hybrid3b_model = clone(clmbr3_model)
ps3b_hybrid = cross_val_predict(
    hybrid3b_model,
    X3b_hybrid,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]

ow3b_curated = overlap_weights(A3b, ps3b_curated)
ow3b_temporal = overlap_weights(A3b, ps3b_temporal)
ow3b_structured = overlap_weights(A3b, ps3b_structured)
ow3b_count = overlap_weights(A3b, ps3b_count)
ow3b_clmbr = overlap_weights(A3b, ps3b_clmbr)
ow3b_hybrid = overlap_weights(A3b, ps3b_hybrid)

# ============================================================
# 4D. BLOCK-PRESERVING OFFSET HYBRID
# ============================================================

lambda_grid_3b = np.array([
    1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0
])
outer_splits3b = list(outer_cv3.split(X3b_curated, A3b))
ps3b_offset = np.full(len(A3b), np.nan)
selected_lambda_3b = []

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_splits3b,
    start=1,
):
    print(f'Offset hybrid outer fold {outer_fold}/5')
    Xc_train = X3b_curated[train_idx]
    Xc_test = X3b_curated[test_idx]
    Xf_train = X3_clmbr_anchored[train_idx]
    Xf_test = X3_clmbr_anchored[test_idx]
    y_train = A3b[train_idx]

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=4200 + outer_fold,
    )
    lambda_losses = {lam: [] for lam in lambda_grid_3b}

    for inner_train_rel, inner_val_rel in inner_cv.split(
        Xc_train,
        y_train,
    ):
        curated_inner = clone(curated3_model)
        curated_inner.fit(
            Xc_train[inner_train_rel],
            y_train[inner_train_rel],
        )
        offset_inner_train = model_logit(
            curated_inner,
            Xc_train[inner_train_rel],
        )
        offset_inner_val = model_logit(
            curated_inner,
            Xc_train[inner_val_rel],
        )

        scaler_inner = StandardScaler()
        Xfi_train = scaler_inner.fit_transform(
            Xf_train[inner_train_rel]
        )
        Xfi_val = scaler_inner.transform(
            Xf_train[inner_val_rel]
        )

        warm_theta = None
        for lam in sorted(lambda_grid_3b, reverse=True):
            theta = fit_offset_ridge(
                Xfi_train,
                y_train[inner_train_rel],
                offset_inner_train,
                lam,
                init=warm_theta,
            )
            p_val = predict_offset_model(
                Xfi_val,
                offset_inner_val,
                theta,
            )
            lambda_losses[lam].append(
                log_loss(
                    y_train[inner_val_rel],
                    p_val,
                    labels=[0, 1],
                )
            )
            warm_theta = theta

    mean_losses = {
        lam: np.mean(lambda_losses[lam])
        for lam in lambda_grid_3b
    }
    best_lambda = min(mean_losses, key=mean_losses.get)
    selected_lambda_3b.append(best_lambda)

    curated_outer = clone(curated3_model)
    curated_outer.fit(Xc_train, y_train)
    offset_outer_train = model_logit(
        curated_outer,
        Xc_train,
    )
    offset_outer_test = model_logit(
        curated_outer,
        Xc_test,
    )

    scaler_outer = StandardScaler()
    Xfo_train = scaler_outer.fit_transform(Xf_train)
    Xfo_test = scaler_outer.transform(Xf_test)
    theta_outer = fit_offset_ridge(
        Xfo_train,
        y_train,
        offset_outer_train,
        best_lambda,
    )
    ps3b_offset[test_idx] = predict_offset_model(
        Xfo_test,
        offset_outer_test,
        theta_outer,
    )

if not np.isfinite(ps3b_offset).all():
    raise RuntimeError('Offset-hybrid cross-fitted propensity scores contain non-finite values.')
ow3b_offset = overlap_weights(A3b, ps3b_offset)
print('Chosen offset lambdas:', selected_lambda_3b)

# ============================================================
# 4E. NAIVE-HYBRID C-GRID SENSITIVITY
# ============================================================
# The primary naive hybrid deliberately uses the same C grid as CLMBR.
# To check whether its result is merely an artifact of that grid, fit a
# broader sensitivity grid while keeping the same five outer folds.

NAIVE_HYBRID_WIDE_C_GRID = [
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
    0.1, 0.3, 1.0, 3.0, 10.0,
]
hybrid3b_wide_model = Pipeline([
    ('scale', StandardScaler()),
    ('logit', LogisticRegressionCV(
        Cs=NAIVE_HYBRID_WIDE_C_GRID,
        cv=3,
        penalty='l2',
        solver='lbfgs',
        scoring='neg_log_loss',
        max_iter=5000,
        n_jobs=-1,
    )),
])
ps3b_hybrid_wide = cross_val_predict(
    hybrid3b_wide_model,
    X3b_hybrid,
    A3b,
    cv=cv3b,
    method='predict_proba',
    n_jobs=-1,
)[:, 1]
ow3b_hybrid_wide = overlap_weights(A3b, ps3b_hybrid_wide)

naive_hybrid_sensitivity = pd.DataFrame([
    {
        'Specification': 'Primary CLMBR C grid',
        'OOF_AUC': roc_auc_score(A3b, ps3b_hybrid),
        'OOF_logloss': log_loss(A3b, ps3b_hybrid, labels=[0, 1]),
        'Estimate_no_noise': weighted_effect(
            y_expected_3b, A3b, ow3b_hybrid
        ),
    },
    {
        'Specification': 'Wider C-grid sensitivity',
        'OOF_AUC': roc_auc_score(A3b, ps3b_hybrid_wide),
        'OOF_logloss': log_loss(A3b, ps3b_hybrid_wide, labels=[0, 1]),
        'Estimate_no_noise': weighted_effect(
            y_expected_3b, A3b, ow3b_hybrid_wide
        ),
    },
])
naive_hybrid_sensitivity['Abs_bias'] = (
    naive_hybrid_sensitivity['Estimate_no_noise'] - TRUE_TAU_3B
).abs()
show(naive_hybrid_sensitivity, 4)
save_table(
    naive_hybrid_sensitivity,
    'cell3B_naive_hybrid_grid_sensitivity.csv',
)

# ============================================================
# 4F. NO-NOISE EFFECT RECOVERY + FIXED-SCALE TRUTH BALANCE
# ============================================================

methods3b = {
    'Oracle': (ps3b_oracle, ow3b_oracle),
    'Clinical + Temporal': (ps3b_structured, ow3b_structured),
    'Offset Hybrid': (ps3b_offset, ow3b_offset),
    'Curated': (ps3b_curated, ow3b_curated),
    'Explicit Temporal': (ps3b_temporal, ow3b_temporal),
    'Naive Hybrid': (ps3b_hybrid, ow3b_hybrid),
    'CLMBR': (ps3b_clmbr, ow3b_clmbr),
    'Count': (ps3b_count, ow3b_count),
    'Crude': (None, np.ones(len(A3b))),
}

truth_clinical = np.column_stack([
    z_sbp, z_dbp, z_bmi, z_creatinine
])
truth_temporal = np.column_stack([
    truth_balance3[c].to_numpy()
    for c in temporal_vars3b
])

rows3b = []
balance_rows3b = []
crude_bias3b = abs(
    weighted_effect(
        y_expected_3b,
        A3b,
        np.ones(len(A3b)),
    )
    - TRUE_TAU_3B
)

for method, (ps, w) in methods3b.items():
    est = weighted_effect(y_expected_3b, A3b, w)
    bias = abs(est - TRUE_TAU_3B)
    clin_smd = np.abs([
        weighted_smd(truth_clinical[:, j], A3b, w)
        for j in range(truth_clinical.shape[1])
    ])
    temp_smd = np.abs([
        weighted_smd(truth_temporal[:, j], A3b, w)
        for j in range(truth_temporal.shape[1])
    ])

    rows3b.append({
        'Method': method,
        'OOF_AUC': np.nan if ps is None else roc_auc_score(A3b, ps),
        'OOF_logloss': np.nan if ps is None else log_loss(A3b, ps, labels=[0, 1]),
        'Estimate': est,
        'Bias': bias,
        'Bias_reduction_pct': (
            100.0 * (crude_bias3b - bias) / crude_bias3b
        ),
        'Clinical_max_SMD': clin_smd.max(),
        'Temporal_max_SMD': temp_smd.max(),
    })

    for block, values in [
        ('Clinical', clin_smd),
        ('Temporal', temp_smd),
    ]:
        balance_rows3b.append({
            'Method': method,
            'Block': block,
            'Mean_abs_SMD': values.mean(),
            'Max_abs_SMD': values.max(),
        })

cell3b = pd.DataFrame(rows3b).sort_values('Bias').reset_index(drop=True)
cell3b_balance = pd.DataFrame(balance_rows3b)

show(cell3b, 4)
show(cell3b_balance, 4)
save_table(cell3b, 'cell3B_summary.csv')
save_table(cell3b_balance, 'cell3B_truth_block_balance.csv')

reference3b = {
    'Oracle': -4.883,
    'Clinical + Temporal': -4.728,
    'Offset Hybrid': -3.606,
    'Curated': -3.533,
    'Explicit Temporal': -3.206,
    'Naive Hybrid': -3.101,
    'CLMBR': -2.431,
    'Count': -2.117,
    'Crude': -2.000,
}
for method, target in reference3b.items():
    got = float(
        cell3b.loc[cell3b['Method'].eq(method), 'Estimate'].iloc[0]
    )
    print(
        f'{method:22s}: computed={got:+.3f}, '
        f'reference={target:+.3f}, delta={got-target:+.3f}'
    )

# ============================================================
# 4G. 500 MIXED-CONFOUNDING OUTCOME SIMULATIONS
# ============================================================
# Treatment assignment and all propensity weights remain fixed.
# The same outcome-noise realization is shared across methods within
# each simulation, making method-to-method differences paired.

N_SIM_3B = 500
NOISE_SD_3B = 2.0
OUTCOME_SEED_3B = 20260821
rng_outcome_3b = np.random.default_rng(OUTCOME_SEED_3B)

mixed_sim_rows = []
for sim_id in range(N_SIM_3B):
    epsilon = rng_outcome_3b.normal(
        0.0,
        NOISE_SD_3B,
        size=len(A3b),
    )
    y = mu0_3b + TRUE_TAU_3B * A3b + epsilon
    for method, (_, w) in methods3b.items():
        mixed_sim_rows.append({
            'simulation': sim_id,
            'method': method,
            'estimate': weighted_effect(y, A3b, w),
        })

cell3b_simulations = pd.DataFrame(mixed_sim_rows)
cell3b_simulations['bias'] = (
    cell3b_simulations['estimate'] - TRUE_TAU_3B
)
cell3b_simulations['squared_error'] = (
    cell3b_simulations['bias'] ** 2
)
cell3b_simulation_summary = (
    cell3b_simulations
    .groupby('method')
    .agg(
        mean_estimate=('estimate', 'mean'),
        mean_bias=('bias', 'mean'),
        MC_SD=('estimate', 'std'),
        RMSE=('squared_error', lambda x: np.sqrt(x.mean())),
    )
    .reset_index()
)
cell3b_simulation_summary['abs_bias'] = (
    cell3b_simulation_summary['mean_bias'].abs()
)
crude_abs_3b = float(
    cell3b_simulation_summary.loc[
        cell3b_simulation_summary['method'].eq('Crude'),
        'abs_bias',
    ].iloc[0]
)
cell3b_simulation_summary['bias_reduction_vs_crude_pct'] = (
    100.0
    * (crude_abs_3b - cell3b_simulation_summary['abs_bias'])
    / crude_abs_3b
)
cell3b_simulation_summary = cell3b_simulation_summary.sort_values(
    'abs_bias'
).reset_index(drop=True)

show(cell3b_simulation_summary, 4)
save_table(cell3b_simulations, 'cell3B_500_simulations.csv')
save_table(cell3b_simulation_summary, 'cell3B_simulation_summary.csv')

# Paired Monte Carlo comparison for the integration finding.
mixed_pivot = cell3b_simulations.pivot(
    index='simulation',
    columns='method',
    values='estimate',
)
offset_minus_curated = (
    mixed_pivot['Offset Hybrid'] - mixed_pivot['Curated']
)
offset_abs_bias_improvement = (
    (mixed_pivot['Curated'] - TRUE_TAU_3B).abs()
    - (mixed_pivot['Offset Hybrid'] - TRUE_TAU_3B).abs()
)

offset_vs_curated_mc = pd.DataFrame([{
    'mean_estimate_delta_offset_minus_curated':
        offset_minus_curated.mean(),
    'SD_estimate_delta':
        offset_minus_curated.std(),
    'mean_abs_bias_improvement':
        offset_abs_bias_improvement.mean(),
    'SD_abs_bias_improvement':
        offset_abs_bias_improvement.std(),
    'fraction_simulations_offset_lower_abs_bias':
        np.mean(offset_abs_bias_improvement > 0),
}])
show(offset_vs_curated_mc, 4)
save_table(
    offset_vs_curated_mc,
    'cell3B_offset_vs_curated_mc.csv',
)
print('\n=== 5. BLINDED ACTIVE-COMPARATOR SCREEN ===')

# ============================================================
# 5A. REBUILD RXNORM DRUG-CLASS VOCABULARIES FROM RXNAV
#     NO SAVED VOCABULARY CACHE IS USED
# ============================================================

DRUG_CLASS_NAMES = {
    'ACE': ['lisinopril','enalapril','ramipril','benazepril','captopril','fosinopril','quinapril','perindopril','trandolapril','moexipril'],
    'ARB': ['losartan','valsartan','irbesartan','candesartan','olmesartan','telmisartan','eprosartan','azilsartan'],
    'GLP1': ['semaglutide','dulaglutide','liraglutide','exenatide','lixisenatide','albiglutide'],
    'DPP4': ['sitagliptin','linagliptin','saxagliptin','alogliptin'],
    'SGLT2': ['canagliflozin','dapagliflozin','empagliflozin','ertugliflozin','bexagliflozin'],
    'METFORMIN': ['metformin'],
    'SU': ['glipizide','glimepiride','glyburide','tolbutamide','tolazamide','chlorpropamide'],
}


def rxnorm_related_codes(name):
    r = requests.get('https://rxnav.nlm.nih.gov/REST/rxcui.json', params={'name': name}, timeout=30)
    r.raise_for_status()
    ids = r.json().get('idGroup', {}).get('rxnormId', [])
    out = set()
    for rxcui in ids:
        rr = requests.get(f'https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json', timeout=30)
        rr.raise_for_status()
        for group in rr.json().get('allRelatedGroup', {}).get('conceptGroup', []) or []:
            for prop in group.get('conceptProperties', []) or []:
                out.add('RxNorm/' + str(prop['rxcui']))
        time.sleep(0.02)
    return out

class_codes = {}
for cls, names in DRUG_CLASS_NAMES.items():
    codes_cls = set()
    print('Resolving', cls)
    for name in names:
        codes_cls |= rxnorm_related_codes(name)
    class_codes[cls] = codes_cls

for cls, vals in class_codes.items():
    print(f'{cls:10s}: {len(vals):4d} RxNorm concepts')

# ============================================================
# 5B. CANDIDATE COHORT + CURATED FEATURE BUILDERS
# ============================================================

PAIR_SPECS = [
    ('ACE', 'ARB', False),
    ('GLP1', 'DPP4', True),
    ('SGLT2', 'DPP4', True),
    ('GLP1', 'SGLT2', True),
    ('DPP4', 'SU', True),
    ('SGLT2', 'SU', True),
    ('METFORMIN', 'SU', True),
]

all_treatment_codes = sorted(set().union(*class_codes.values()))
drug_events = ehr_ds.to_table(
    columns=['subject_id', 'time', 'code'],
    filter=(ds.field('omop_table') == 'drug_exposure') & ds.field('code').isin(all_treatment_codes),
).to_pandas()
drug_events['time'] = pd.to_datetime(drug_events['time'])

class_events = {
    cls: drug_events.loc[drug_events['code'].isin(codes_cls), ['subject_id','time','code']].copy()
    for cls, codes_cls in class_codes.items()
}


def build_first_of_pair(left, right):
    l = class_events[left].groupby('subject_id')['time'].min().rename('left_time')
    r = class_events[right].groupby('subject_id')['time'].min().rename('right_time')
    c = pd.concat([l, r], axis=1)
    c = c.loc[c[['left_time','right_time']].notna().any(axis=1)].copy()
    same_day = c['left_time'].notna() & c['right_time'].notna() & (
        c['left_time'].dt.normalize() == c['right_time'].dt.normalize()
    )
    c = c.loc[~same_day].copy()
    left_first = c['right_time'].isna() | (c['left_time'].notna() & (c['left_time'] < c['right_time']))
    c['treatment'] = np.where(left_first, left, right)
    c['A'] = c['treatment'].eq(right).astype(int)
    c['index_time'] = c[['left_time','right_time']].min(axis=1)
    return c.reset_index()[['subject_id','treatment','A','index_time']]

provisional = {f'{l} vs {r}': build_first_of_pair(l, r) for l, r, _ in PAIR_SPECS}
union_ids = sorted(set().union(*[set(x['subject_id']) for x in provisional.values()]))

screen_raw = ehr_ds.to_table(
    columns=['subject_id','time','code','numeric_value','text_value','unit','omop_table'],
    filter=ds.field('subject_id').isin(union_ids),
).to_pandas()
screen_raw['time'] = pd.to_datetime(screen_raw['time'])
nonperson = screen_raw.loc[screen_raw['omop_table'] != 'person']
first_clinical = nonperson.groupby('subject_id')['time'].min().rename('first_clinical')
birth_date = screen_raw.loc[screen_raw['omop_table'].eq('person')].groupby('subject_id')['time'].min().rename('birth_date')

T2D_CODE = 'SNOMED/44054006'
HBA1C_CODE = 'LOINC/4548-4'


def apply_pair_eligibility(cohort, diabetes_pair):
    c = cohort.merge(first_clinical, on='subject_id', how='left')
    c['history_days'] = (c['index_time'] - c['first_clinical']).dt.total_seconds() / 86400.0
    c = c.loc[c['history_days'] >= 365].copy()
    if not diabetes_pair:
        return c

    d = screen_raw.loc[screen_raw['subject_id'].isin(c['subject_id'])].merge(
        c[['subject_id','index_time']], on='subject_id', how='inner'
    )
    d = d.loc[d['time'] < d['index_time']].copy()
    t2d_ids = set(d.loc[d['code'].eq(T2D_CODE), 'subject_id'])
    hba = pd.to_numeric(d['numeric_value'], errors='coerce')
    hba_ids = set(d.loc[d['code'].eq(HBA1C_CODE) & (hba >= 6.5), 'subject_id'])
    diabetes_med_codes = set().union(
        class_codes['METFORMIN'], class_codes['SU'], class_codes['DPP4'], class_codes['SGLT2'], class_codes['GLP1']
    )
    prior_med_ids = set(d.loc[d['code'].isin(diabetes_med_codes), 'subject_id'])
    return c.loc[c['subject_id'].isin(t2d_ids | hba_ids | prior_med_ids)].copy()

COUNT_TABLES = ['condition_occurrence','drug_exposure','measurement','procedure_occurrence','visit_occurrence']


def make_pair_features(cohort):
    d = screen_raw.loc[screen_raw['subject_id'].isin(cohort['subject_id'])].merge(
        cohort[['subject_id','index_time']], on='subject_id', how='inner'
    )
    d['day'] = (d['time'] - d['index_time']).dt.total_seconds() / 86400.0
    baseline = d.loc[(d['day'] >= -365) & (d['day'] < 0) & ~d['omop_table'].eq('person')].copy()
    f = cohort[['subject_id','index_time','A']].copy().merge(birth_date, on='subject_id', how='left')
    f['age'] = (f['index_time'] - f['birth_date']).dt.days / 365.25
    f['index_year'] = f['index_time'].dt.year

    overall = baseline.groupby('subject_id').agg(
        n_events=('code','size'), n_codes=('code','nunique'),
        n_dates=('time', lambda x: x.dt.normalize().nunique()),
    ).reset_index()
    f = f.merge(overall, on='subject_id', how='left')

    table_counts = baseline.loc[baseline['omop_table'].isin(COUNT_TABLES)].groupby(
        ['subject_id','omop_table']
    ).size().unstack(fill_value=0)
    for t in COUNT_TABLES:
        if t not in table_counts.columns:
            table_counts[t] = 0
    table_counts = table_counts[COUNT_TABLES].add_prefix('n_').reset_index()
    f = f.merge(table_counts, on='subject_id', how='left')

    for loinc, name in [('LOINC/39156-5','BMI'), ('LOINC/4548-4','HbA1c')]:
        m = baseline.loc[baseline['code'].eq(loinc), ['subject_id','time','numeric_value']].copy()
        m['numeric_value'] = pd.to_numeric(m['numeric_value'], errors='coerce')
        m = m.dropna(subset=['numeric_value']).sort_values('time').groupby('subject_id').tail(1)[['subject_id','numeric_value']]
        f = f.merge(m.rename(columns={'numeric_value': name}), on='subject_id', how='left')

    for cls, codes_cls in class_codes.items():
        prior_ids = set(baseline.loc[baseline['code'].isin(codes_cls), 'subject_id'])
        f['prior_' + cls] = f['subject_id'].isin(prior_ids).astype(int)

    count_cols = ['n_events','n_codes','n_dates'] + ['n_' + x for x in COUNT_TABLES]
    for ccol in count_cols:
        if ccol not in f:
            f[ccol] = 0
        f[ccol] = pd.to_numeric(f[ccol], errors='coerce').fillna(0)
        f['log_' + ccol] = np.log1p(f[ccol])

    features = ['age','index_year','BMI','HbA1c'] + ['log_' + c for c in count_cols] + ['prior_' + cls for cls in class_codes]
    return f, features

# ============================================================
# 5C. OUTCOME-AVAILABILITY COUNTS + FIRST-STAGE SCREEN
# ============================================================


def endpoint_counts(cohort):
    c = screen_raw.loc[
        screen_raw['subject_id'].isin(cohort['subject_id']) & screen_raw['omop_table'].eq('condition_occurrence'),
        ['subject_id','time','code'],
    ].merge(cohort[['subject_id','index_time','A']], on='subject_id', how='inner')
    c['day'] = (c['time'] - c['index_time']).dt.total_seconds() / 86400.0
    base = c.loc[(c['day'] >= -365) & (c['day'] < 0), ['subject_id','code']].drop_duplicates()
    post = c.loc[(c['day'] > 0) & (c['day'] <= 365), ['subject_id','code','A']].drop_duplicates(['subject_id','code'])
    base['_seen'] = 1
    inc = post.merge(base, on=['subject_id','code'], how='left').loc[lambda x: x['_seen'].isna()]
    counts = inc.groupby(['code','A'])['subject_id'].nunique().unstack(fill_value=0)
    for a in [0, 1]:
        if a not in counts.columns:
            counts[a] = 0
    counts['total'] = counts[0] + counts[1]
    counts['min_arm'] = counts[[0,1]].min(axis=1)
    return int(((counts['total'] >= 20) & (counts['min_arm'] >= 5)).sum()), int(((counts['total'] >= 10) & (counts['min_arm'] >= 3)).sum())

MIN_ARM_N = 100
MIN_AUC = 0.65
MIN_ESS_RATIO = 0.60
MAX_EXTREME_PS_FRAC = 0.05
MIN_ENDPOINTS = 10

stage1_rows = []
eligible_pair_cohorts = {}
for i, (left, right, diabetes_pair) in enumerate(PAIR_SPECS, start=1):
    pair = f'{left} vs {right}'
    cohort = apply_pair_eligibility(provisional[pair], diabetes_pair).sort_values('subject_id').reset_index(drop=True)
    eligible_pair_cohorts[pair] = cohort
    n0, n1 = int((cohort['A'] == 0).sum()), int((cohort['A'] == 1).sum())
    if min(n0, n1) < 15:
        stage1_rows.append({'Pair': pair, 'N_left': n0, 'N_right': n1, 'Min_arm_N': min(n0,n1),
                            'Curated_OOF_AUC': np.nan, 'Extreme_PS_frac': np.nan, 'Min_ESS_ratio': np.nan,
                            'Endpoints_20_5': 0, 'Endpoints_10_3': 0})
        continue
    f, cols = make_pair_features(cohort)
    f = f.set_index('subject_id').loc[cohort['subject_id']].reset_index()
    X = f[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    y_pair = cohort['A'].to_numpy(int)
    res = nested_oof_logistic(X, y_pair, 20260819 + i, C_grid=(0.03,0.1,0.3,1.0,3.0))
    ps = res['ps']; ow = overlap_weights(y_pair, ps)
    min_ess = min(ess(ow[y_pair == 0]) / n0, ess(ow[y_pair == 1]) / n1)
    extreme = np.mean((ps < 0.05) | (ps > 0.95))
    n20, n10 = endpoint_counts(cohort)
    stage1_rows.append({'Pair': pair, 'N_left': n0, 'N_right': n1, 'Min_arm_N': min(n0,n1),
                        'Curated_OOF_AUC': res['auc'], 'Extreme_PS_frac': extreme, 'Min_ESS_ratio': min_ess,
                        'Endpoints_20_5': n20, 'Endpoints_10_3': n10})

comparator_screen = pd.DataFrame(stage1_rows)
comparator_screen['Pass_N'] = comparator_screen['Min_arm_N'] >= MIN_ARM_N
comparator_screen['Pass_AUC'] = comparator_screen['Curated_OOF_AUC'] >= MIN_AUC
comparator_screen['Pass_overlap'] = (comparator_screen['Min_ESS_ratio'] >= MIN_ESS_RATIO) & (comparator_screen['Extreme_PS_frac'] <= MAX_EXTREME_PS_FRAC)
comparator_screen['Pass_outcomes'] = comparator_screen['Endpoints_20_5'] >= MIN_ENDPOINTS
comparator_screen['QUALIFIES'] = comparator_screen[['Pass_N','Pass_AUC','Pass_overlap','Pass_outcomes']].all(axis=1)
comparator_screen['Criteria_passed'] = comparator_screen[['Pass_N','Pass_AUC','Pass_overlap','Pass_outcomes']].sum(axis=1)
# Preserve the first-stage ranking because this order is also used to assign second-stage CV seeds.
comparator_screen = comparator_screen.sort_values(
    ['QUALIFIES','Criteria_passed','Endpoints_20_5','Curated_OOF_AUC'],
    ascending=[False,False,False,False]
).reset_index(drop=True)

show(comparator_screen, 3)
save_table(comparator_screen, 'active_comparator_screen.csv')
print('First-stage qualifying pairs:', comparator_screen.loc[comparator_screen['QUALIFIES'], 'Pair'].tolist())
print('CLMBR has not been used.')

print('\n=== 5. TEMPORAL-SIGNAL PRESELECTION ===')

# ============================================================
# 6A. EXPLICIT TEMPORAL FEATURE BUILDER
# ============================================================

TEMP_TABLES = ['condition_occurrence','drug_exposure','measurement','procedure_occurrence','visit_occurrence']
RECENT_DAYS = 90.0
OLDER_DAYS = 275.0
OLDER_TO_90 = RECENT_DAYS / OLDER_DAYS


def build_temporal_features(cohort):
    d = screen_raw.loc[screen_raw['subject_id'].isin(cohort['subject_id'])].merge(
        cohort[['subject_id','index_time','A']], on='subject_id', how='inner'
    )
    d['day'] = (d['time'] - d['index_time']).dt.total_seconds() / 86400.0
    recent = d.loc[(d['day'] >= -90) & (d['day'] < 0) & ~d['omop_table'].eq('person')].copy()
    older = d.loc[(d['day'] >= -365) & (d['day'] < -90) & ~d['omop_table'].eq('person')].copy()
    out = cohort[['subject_id','A']].copy()

    def attach_count_change(recent_df, older_df, prefix, distinct_dates=False):
        if distinct_dates:
            rc = recent_df.assign(_date=recent_df['time'].dt.normalize()).groupby('subject_id')['_date'].nunique()
            oc = older_df.assign(_date=older_df['time'].dt.normalize()).groupby('subject_id')['_date'].nunique()
        else:
            rc = recent_df.groupby('subject_id').size()
            oc = older_df.groupby('subject_id').size()
        tmp = pd.concat([rc.rename(prefix+'_recent_raw'), oc.rename(prefix+'_older_raw')], axis=1).fillna(0).reset_index()
        tmp[prefix+'_recent_90eq'] = tmp[prefix+'_recent_raw']
        tmp[prefix+'_older_90eq'] = tmp[prefix+'_older_raw'] * OLDER_TO_90
        tmp[prefix+'_log_recent'] = np.log1p(tmp[prefix+'_recent_90eq'])
        tmp[prefix+'_log_older'] = np.log1p(tmp[prefix+'_older_90eq'])
        tmp[prefix+'_change'] = tmp[prefix+'_log_recent'] - tmp[prefix+'_log_older']
        return tmp[['subject_id', prefix+'_log_recent', prefix+'_log_older', prefix+'_change']]

    out = out.merge(attach_count_change(recent, older, 'all_events'), on='subject_id', how='left')
    out = out.merge(attach_count_change(recent, older, 'dates', distinct_dates=True), on='subject_id', how='left')
    for table in TEMP_TABLES:
        out = out.merge(
            attach_count_change(recent.loc[recent['omop_table'].eq(table)], older.loc[older['omop_table'].eq(table)], table),
            on='subject_id', how='left'
        )

    for code_loinc, label in [('LOINC/39156-5','BMI'), ('LOINC/4548-4','HbA1c')]:
        def last_value(df):
            x = df.loc[df['code'].eq(code_loinc), ['subject_id','time','numeric_value']].copy()
            x['numeric_value'] = pd.to_numeric(x['numeric_value'], errors='coerce')
            return x.dropna(subset=['numeric_value']).sort_values('time').groupby('subject_id').tail(1)[['subject_id','numeric_value']]
        rv = last_value(recent).rename(columns={'numeric_value': label+'_recent'})
        ov = last_value(older).rename(columns={'numeric_value': label+'_older'})
        out = out.merge(rv, on='subject_id', how='left').merge(ov, on='subject_id', how='left')
        out[label+'_change'] = out[label+'_recent'] - out[label+'_older']
        out[label+'_recent_present'] = out[label+'_recent'].notna().astype(int)
        out[label+'_older_present'] = out[label+'_older'].notna().astype(int)

    count_like = [c for c in out.columns if ('log_recent' in c or 'log_older' in c or (c.endswith('_change') and not c.startswith('BMI') and not c.startswith('HbA1c')))]
    out[count_like] = out[count_like].fillna(0)
    temporal_cols = [c for c in out.columns if c not in ['subject_id','A']]
    return out, temporal_cols

# ============================================================
# 6B. RUN TEMPORAL-SIGNAL SCREEN (NO CLMBR, NO OUTCOME EFFECTS)
# ============================================================

TEMP_SCREEN_MIN_ARM_N = 50
TEMP_SCREEN_MIN_ESS = 0.60
TEMP_SCREEN_MAX_EXTREME = 0.05
TEMP_SCREEN_MIN_ENDPOINTS_10_3 = 10

temporal_rows = []
# Iterate over the ranked first-stage table; pair_i is the 0-based ranked row index and enters the CV seed.
for pair_i, row in comparator_screen.iterrows():
    pair = row['Pair']
    cohort = eligible_pair_cohorts[pair].copy().sort_values('subject_id').reset_index(drop=True)
    n0, n1 = int((cohort['A'] == 0).sum()), int((cohort['A'] == 1).sum())

    cur_df, cur_cols = make_pair_features(cohort)
    cur_df = cur_df.set_index('subject_id').loc[cohort['subject_id']].reset_index()
    temp_df, temp_cols = build_temporal_features(cohort)
    temp_df = temp_df.set_index('subject_id').loc[cohort['subject_id']].reset_index()

    y_pair = cohort['A'].to_numpy(dtype=int)
    Xc = cur_df[cur_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    Xt = temp_df[temp_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    Xct = np.column_stack([Xc, Xt])

    # SAME outer seed for curated and curated+temporal => identical outer folds.
    seed = 20261000 + pair_i
    res_cur = nested_oof_logistic(Xc, y_pair, seed)
    res_temp = nested_oof_logistic(Xct, y_pair, seed)

    delta_auc = res_temp['auc'] - res_cur['auc']
    delta_ll = res_cur['logloss'] - res_temp['logloss']  # positive = improvement
    rel_ll = 100 * delta_ll / res_cur['logloss']

    eligible = (
        min(n0, n1) >= TEMP_SCREEN_MIN_ARM_N
        and row['Min_ESS_ratio'] >= TEMP_SCREEN_MIN_ESS
        and row['Extreme_PS_frac'] <= TEMP_SCREEN_MAX_EXTREME
        and row['Endpoints_10_3'] >= TEMP_SCREEN_MIN_ENDPOINTS_10_3
    )

    temporal_rows.append({
        'Pair': pair,
        'N_left': n0,
        'N_right': n1,
        'Min_arm_N': min(n0, n1),
        'Endpoints_20_5': int(row['Endpoints_20_5']),
        'Endpoints_10_3': int(row['Endpoints_10_3']),
        'Previous_min_ESS_ratio': row['Min_ESS_ratio'],
        'Previous_extreme_PS_frac': row['Extreme_PS_frac'],
        'Curated_AUC': res_cur['auc'],
        'Curated_Temporal_AUC': res_temp['auc'],
        'Delta_AUC': delta_auc,
        'Curated_logloss': res_cur['logloss'],
        'Curated_Temporal_logloss': res_temp['logloss'],
        'Delta_logloss': delta_ll,
        'Relative_logloss_improvement_pct': rel_ll,
        'SECOND_STAGE_ELIGIBLE': eligible,
    })

temporal_comparator_screen = pd.DataFrame(temporal_rows).sort_values(
    ['SECOND_STAGE_ELIGIBLE','Delta_logloss','Delta_AUC'], ascending=[False,False,False]
).reset_index(drop=True)
show(temporal_comparator_screen, 4)

eligible_temporal = temporal_comparator_screen.loc[temporal_comparator_screen['SECOND_STAGE_ELIGIBLE']]
TEMPORAL_SCREEN_WINNER = None if eligible_temporal.empty else eligible_temporal.iloc[0]['Pair']
save_table(temporal_comparator_screen, 'temporal_signal_screen.csv')
print('Temporal-signal screen winner:', TEMPORAL_SCREEN_WINNER)
print('Still no CLMBR and no post-treatment effect estimates have been examined.')
REFERENCE_TEMPORAL_SCREEN_WINNER = 'METFORMIN vs SU'
if TEMPORAL_SCREEN_WINNER != REFERENCE_TEMPORAL_SCREEN_WINNER:
    warnings.warn(
        'Temporal-screen winner differs from the reference run: '
        f'observed={TEMPORAL_SCREEN_WINNER!r}, '
        f'reference={REFERENCE_TEMPORAL_SCREEN_WINNER!r}. '
        'The reported Metformin vs SU analysis is retained as the prespecified '
        'real-data analysis and is not relabeled as the winner from this rerun.'
    )

print('\n=== 7. FROZEN REFERENCE REPLICATION: METFORMIN vs SU REAL TREATMENT MODEL ===')

# ============================================================
# 7A. FREEZE SELECTED COHORT + COMPUTE ITS CLMBR MATRIX FRESH
# ============================================================

metsu = eligible_pair_cohorts['METFORMIN vs SU'].sort_values('subject_id').reset_index(drop=True)
print(metsu['treatment'].value_counts())
if len(metsu) != 236:
    warnings.warn(
        f'Reference Metformin/SU cohort size was 236; this run produced '
        f'{len(metsu)}. Exact values may differ if the data release or '
        'RxNorm vocabulary has changed.'
    )

X_metsu_clmbr, metsu_embedding_meta = extract_clmbr_matrix(
    metsu[['subject_id', 'index_time']],
    'index_time',
    window_days=365,
)
metsu_embedding_meta = (
    metsu_embedding_meta.set_index('subject_id')
    .loc[metsu['subject_id']]
    .reset_index()
)
if X_metsu_clmbr.shape != (len(metsu), 768):
    raise RuntimeError(
        f'Expected Metformin/SU CLMBR shape {(len(metsu), 768)}, '
        f'found {X_metsu_clmbr.shape}.'
    )
print('Embedding shape:', X_metsu_clmbr.shape)
print(
    'Median representation gap:',
    round(metsu_embedding_meta['representation_gap_days'].median(), 2),
    'days',
)

# ============================================================
# 7B. BUILD THE THREE INPUT REPRESENTATIONS
# ============================================================

y = metsu['A'].to_numpy(int)
cur_df, cur_cols = make_pair_features(metsu)
cur_df = cur_df.set_index('subject_id').loc[metsu['subject_id']].reset_index()
temp_df, temp_cols = build_temporal_features(metsu)
temp_df = temp_df.set_index('subject_id').loc[metsu['subject_id']].reset_index()

X_cur = cur_df[cur_cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
X_temp = temp_df[temp_cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
X_cur_temp = np.column_stack([X_cur, X_temp])
X_clm = np.asarray(X_metsu_clmbr, dtype=float)

print('Curated:', X_cur.shape)
print('Temporal:', X_temp.shape)
print('Curated + Temporal:', X_cur_temp.shape)
print('CLMBR:', X_clm.shape)

# ============================================================
# 7C. CROSS-FIT CURATED / TEMPORAL / CLMBR / OFFSET HYBRID
#
# Cross-fitting specification:
#   * same 5 outer folds for every method
#   * inner 3-fold log-loss tuning only inside outer training data
#   * fixed seeds and hyperparameter grids
#   * offset hybrid keeps the curated logit fixed and lets CLMBR explain residual assignment
# ============================================================

OUTER_SEED = 20260819
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
outer_splits = list(outer_cv.split(X_cur, y))

STANDARD_C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CLMBR_C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
OFFSET_LAMBDA_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

N = len(y)
ps_cur = np.full(N, np.nan)
ps_cur_temp = np.full(N, np.nan)
ps_clm = np.full(N, np.nan)
ps_offset = np.full(N, np.nan)
selected_cur_C, selected_temp_C, selected_clm_C, selected_lambda = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
    print(f'Fold {fold}/5')
    y_tr = y[train_idx]

    # 6A. Curated
    best_cur_C = select_C(X_cur[train_idx], y_tr, STANDARD_C_GRID, seed=10000 + fold)
    selected_cur_C.append(best_cur_C)
    cur_model = make_logistic(best_cur_C)
    cur_model.fit(X_cur[train_idx], y_tr)
    ps_cur[test_idx] = cur_model.predict_proba(X_cur[test_idx])[:, 1]

    # 6B. Curated + explicit temporal
    best_temp_C = select_C(X_cur_temp[train_idx], y_tr, STANDARD_C_GRID, seed=20000 + fold)
    selected_temp_C.append(best_temp_C)
    temp_model = make_logistic(best_temp_C)
    temp_model.fit(X_cur_temp[train_idx], y_tr)
    ps_cur_temp[test_idx] = temp_model.predict_proba(X_cur_temp[test_idx])[:, 1]

    # 6C. CLMBR alone
    best_clm_C = select_C(X_clm[train_idx], y_tr, CLMBR_C_GRID, seed=30000 + fold)
    selected_clm_C.append(best_clm_C)
    clm_model = make_logistic(best_clm_C)
    clm_model.fit(X_clm[train_idx], y_tr)
    ps_clm[test_idx] = clm_model.predict_proba(X_clm[test_idx])[:, 1]

    # 6D. Block-preserving curated + CLMBR offset
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=40000 + fold)
    lambda_losses = {lam: [] for lam in OFFSET_LAMBDA_GRID}
    Xc_outer = X_cur[train_idx]
    Xf_outer = X_clm[train_idx]

    for inner_tr, inner_va in inner.split(Xc_outer, y_tr):
        # Select/refit curated baseline strictly within inner training.
        inner_cur_C = select_C(
            Xc_outer[inner_tr],
            y_tr[inner_tr],
            STANDARD_C_GRID,
            seed=50000 + 100 * fold + len(inner_tr),
        )
        inner_cur_model = make_logistic(inner_cur_C)
        inner_cur_model.fit(Xc_outer[inner_tr], y_tr[inner_tr])
        offset_tr = model_logit(inner_cur_model, Xc_outer[inner_tr])
        offset_va = model_logit(inner_cur_model, Xc_outer[inner_va])

        scaler = StandardScaler()
        Xf_tr_scaled = scaler.fit_transform(Xf_outer[inner_tr])
        Xf_va_scaled = scaler.transform(Xf_outer[inner_va])

        for lam in OFFSET_LAMBDA_GRID:
            theta = fit_offset_ridge(
                Xf_tr_scaled, y_tr[inner_tr], offset_tr, lam
            )
            pred = predict_offset_model(Xf_va_scaled, offset_va, theta)
            lambda_losses[lam].append(
                log_loss(y_tr[inner_va], pred, labels=[0, 1])
            )

    best_lambda = min(
        OFFSET_LAMBDA_GRID,
        key=lambda lam: np.mean(lambda_losses[lam]),
    )
    selected_lambda.append(best_lambda)

    # Fit the offset hybrid on the complete outer training set.
    curated_offset_train = model_logit(cur_model, X_cur[train_idx])
    curated_offset_test = model_logit(cur_model, X_cur[test_idx])
    fm_scaler = StandardScaler()
    Xfm_train = fm_scaler.fit_transform(X_clm[train_idx])
    Xfm_test = fm_scaler.transform(X_clm[test_idx])
    theta = fit_offset_ridge(
        Xfm_train, y_tr, curated_offset_train, best_lambda
    )
    ps_offset[test_idx] = predict_offset_model(
        Xfm_test, curated_offset_test, theta
    )

for name, ps in {
    'Curated': ps_cur,
    'Curated + Temporal': ps_cur_temp,
    'CLMBR': ps_clm,
    'Curated + CLMBR Offset': ps_offset,
}.items():
    assert np.isfinite(ps).all(), name

curated_logloss = log_loss(y, ps_cur, labels=[0, 1])
real_rows = []
for method, ps in [
    ('Curated', ps_cur),
    ('Curated + Temporal', ps_cur_temp),
    ('CLMBR', ps_clm),
    ('Curated + CLMBR Offset', ps_offset),
]:
    ll = log_loss(y, ps, labels=[0, 1])
    auc = roc_auc_score(y, ps)
    ow = overlap_weights(y, ps)
    d = {
        'Extreme_PS_frac': np.mean((ps < 0.05) | (ps > 0.95)),
        'ESS_Metformin': ess(ow[y == 0]),
        'ESS_SU': ess(ow[y == 1]),
        'Min_ESS_ratio': min(ess(ow[y == 0]) / (y == 0).sum(), ess(ow[y == 1]) / (y == 1).sum()),
    }
    real_rows.append({
        'Method': method,
        'OOF_AUC': auc,
        'OOF_logloss': ll,
        'Delta_AUC_vs_Curated': auc - roc_auc_score(y, ps_cur),
        'Logloss_improvement_vs_Curated_pct': 100 * (curated_logloss - ll) / curated_logloss,
        **d,
    })

real_treatment_model = pd.DataFrame(real_rows)
show(real_treatment_model, 4)
print('Selected Curated C:', selected_cur_C)
print('Selected Temporal C:', selected_temp_C)
print('Selected CLMBR C:', selected_clm_C)
print('Selected Offset lambda:', selected_lambda)

# Reference values for reproducibility checks; never used to create the result table.
reference_real = {
    'Curated': (0.635, 0.6064),
    'Curated + Temporal': (0.648, 0.6067),
    'CLMBR': (0.751, 0.5845),
    'Curated + CLMBR Offset': (0.743, 0.5478),
}
save_table(real_treatment_model, 'metformin_su_real_treatment_model.csv')
save_table(metsu_embedding_meta, 'metformin_su_clmbr_metadata_generated.csv')

for method, (auc_target, ll_target) in reference_real.items():
    r = real_treatment_model.loc[real_treatment_model['Method'].eq(method)].iloc[0]
    print(
        f'{method:25s} computed AUC={r.OOF_AUC:.3f}, logloss={r.OOF_logloss:.4f} | '
        f'reference AUC={auc_target:.3f}, logloss={ll_target:.4f}'
    )

print('\n=== 8. OBSERVED CLINICAL + TEMPORAL BALANCE ===')

# ============================================================
# 8A. OBSERVED COVARIATE BALANCE
# ============================================================

w_curated = overlap_weights(y, ps_cur)
w_temporal = overlap_weights(y, ps_cur_temp)
w_hybrid = overlap_weights(y, ps_offset)

# Median-impute only for balance calculation; imputation is done featurewise on the full descriptive table.
def impute_for_balance(X):
    return SimpleImputer(strategy='median').fit_transform(np.asarray(X, float))

Xc_bal = impute_for_balance(X_cur)
Xt_bal = impute_for_balance(X_temp)
weight_sets = {
    'Unweighted': np.ones(len(y)),
    'Curated': w_curated,
    'Curated + Temporal': w_temporal,
    'Curated + CLMBR Offset': w_hybrid,
}

balance_summary_rows = []
residual_rows = []
for block_name, Xb, names in [('Curated', Xc_bal, cur_cols), ('Temporal', Xt_bal, temp_cols)]:
    for method, w in weight_sets.items():
        smds = np.array([weighted_smd(Xb[:,j], y, w) for j in range(Xb.shape[1])])
        balance_summary_rows.append({
            'Block': block_name, 'Method': method,
            'Mean_abs_SMD': np.mean(np.abs(smds)),
            'Max_abs_SMD': np.max(np.abs(smds)),
            'N_over_010': int((np.abs(smds)>0.10).sum()),
            'N_over_020': int((np.abs(smds)>0.20).sum()),
        })
        if method == 'Curated + CLMBR Offset':
            for feat, smd in zip(names, smds):
                residual_rows.append({'Block': block_name, 'Feature': feat, 'SMD': smd})

balance_summary = pd.DataFrame(balance_summary_rows)
residual_balance = pd.DataFrame(residual_rows).assign(abs_SMD=lambda x: x['SMD'].abs()).sort_values('abs_SMD', ascending=False)
show(balance_summary, 3)
save_table(balance_summary, 'metformin_su_balance_summary.csv')
save_table(residual_balance, 'metformin_su_residual_balance.csv')
print('Largest residual imbalances under FM-augmented adjustment:')
show(residual_balance.head(15), 3)



print('\n============================================================')
print('CONFOUNDING-ONLY REPRODUCTION COMPLETE')
print('============================================================')
print('Analysis complete. Generated tables are in:', OUTPUT_DIR)

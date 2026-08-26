# Data and model access

This repository does not redistribute the source EHR data or pretrained model weights.

## EHRSHOT

The analysis expects the EHRSHOT MEDS release under a directory such as:

```text
/path/to/EHRSHOT/EHRSHOT_MEDS/data/*.parquet
```

Obtain EHRSHOT from the Stanford Shah Lab EHRSHOT benchmark repository and follow its current research-use requirements.

## CLMBR-T-Base

The pretrained model is expected under a directory such as:

```text
/path/to/EHRSHOT/clmbr-t-base/
```

The script can optionally use Hugging Face `snapshot_download` when `--download-clmbr-if-missing` is supplied, but access to `StanfordShahLab/clmbr-t-base` may require an authenticated account and acceptance of the model's access terms.

## RxNav

The active-comparator portion rebuilds RxNorm drug-class vocabularies from the public NIH RxNav API at runtime. A network connection is therefore required for a complete run of that section.

## No source data in this repository

Do not commit patient-level EHRSHOT files, generated patient-level CLMBR matrices, or other source-data derivatives that are restricted by the underlying data-use terms.

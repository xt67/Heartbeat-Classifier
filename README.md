# Heartbeat Classifier — started

Research pipeline for four-class ECG beat classification with a 1D CNN. This is an
educational experiment, not a clinical device. It requires annotated beat locations;
it does not find beats or diagnose an unannotated ECG recording.

From the Heartbeat-Classifier repository root:

```powershell
python -m pip install -r requirements.txt
python -m heartbeat_classifier.data --download
python -m heartbeat_classifier.train --epochs 20
```

Download requires internet. Existing WFDB records can be supplied with `--raw PATH`
without `--download`. No recordings, real-data results or ECG models are bundled.

## Preprocessing

Use MIT-BIH MLII at 360 Hz. Windows contain 90 samples before and 126 samples starting
at each annotation (216 samples, 0.6 seconds). Skip incomplete boundary windows,
nonfinite windows and symbols outside the four mapped groups. Normalize each beat
independently using its own mean/std; no statistics are fit on the test set.

| Class | Annotation symbols |
| --- | --- |
| N: normal group | N, L, R, e, j |
| S: supraventricular | A, a, J, S |
| V: ventricular | V, E |
| F: fusion | F |

Unknown and paced beats are excluded, so this is not classification of every MIT-BIH
beat. Paced records 102, 104, 107 and 217 are excluded.

## Honest split and comparison

The split adapts the conventional DS1/DS2 record lists: 21 training records, records
201/202 for validation, and 21 test records. Records 201/202 are from one patient
and stay together. This is a custom partition, not a directly comparable standard
DS1/DS2 benchmark. The test set is fixed before training.

Matched baseline and class-weighted CNN runs use the same initialization seed,
minibatch order and training budget. Weights are N/(4 × class count), computed on
training labels only. Select each checkpoint by validation macro F1. Evaluate the
selected checkpoint on the held-out test set, reporting per-class recall, precision,
F1, support, accuracy and a confusion matrix (rows=true; columns=predicted).
`comparison.json` reports recall changes in percentage points. Class weighting
is an experiment: an improvement is not guaranteed. A missing test class has null
recall. Macro metrics use the fixed four-class label space.

## Next steps

Run the real download and inspect class counts; train both models; inspect confusion
matrices and rare-class errors; repeat with multiple seeds; then write the resume
line using measured results. A Kaggle beat-level dataset is not a substitute for this
patient split unless patient identifiers are available.

Sources: https://physionet.org/content/mitdb/1.0.0/
and https://github.com/MIT-LCP/wfdb-python

## Verify preprocessing and metrics

```powershell
python -m unittest discover -s tests -v
```

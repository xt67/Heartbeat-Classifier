# Heartbeat verification — 18 November 2026

Tests passed for beat window extraction, label mapping, patient group separation,
normalization and minority recall metrics. Both CNN variants completed a one-epoch
synthetic-data smoke run, saving checkpoints and the comparison report. Those
synthetic results are software checks only and are not ECG performance results.

The real MIT-BIH download, WFDB integration and real-data model evaluation remain
unrun: WFDB is absent locally and the execution environment cannot resolve external
hosts. No clinical accuracy or rare-class improvement is claimed.

Both PyTorch smoke runs completed despite the installed PyTorch 2.2 / NumPy 2.5.4
ABI warning. Requirements use NumPy <2 for a fresh compatible environment.

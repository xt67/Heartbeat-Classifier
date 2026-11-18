"""MIT-BIH preprocessing with record-disjoint, patient-aware partitioning."""
import argparse
import json
from pathlib import Path
import numpy as np

CLASSES = ['N', 'S', 'V', 'F']
LABELS = {symbol: i for i, symbols in enumerate(['NLRej', 'AaJS', 'VE', 'F']) for symbol in symbols}
# Conventional DS1/DS2 partition, excluding paced records 102,104,107,217.
# Records 201 and 202 belong to the same subject: keep both in validation.
TRAIN = '101 106 108 109 112 114 115 116 118 119 122 124 203 205 207 208 209 215 220 223 230'.split()
VALIDATION = ['201', '202']
TEST = '100 103 105 111 113 117 121 123 200 210 212 213 214 219 221 222 228 231 232 233 234'.split()


def windows(signal, samples, symbols, before=90, after=126):
    beats, labels = [], []
    for sample, symbol in zip(samples, symbols):
        if symbol not in LABELS or sample < before or sample+after > len(signal):
            continue
        beat = np.asarray(signal[sample-before:sample+after], dtype=np.float32)
        if not np.isfinite(beat).all():
            continue
        beat = (beat - beat.mean()) / max(float(beat.std()), 1e-6)
        beats.append(beat); labels.append(LABELS[symbol])
    return np.asarray(beats, dtype='f').reshape(-1, before+after), np.asarray(labels, dtype=np.int64)


def main():
    import wfdb
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='heartbeat_classifier/data/raw')
    parser.add_argument('--output', default='heartbeat_classifier/data/beats.npz')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    raw = Path(args.raw)
    if args.download:
        raw.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database('mitdb', str(raw), records=TRAIN+VALIDATION+TEST)
    arrays, counts = {}, {}
    for split, records in [('train', TRAIN), ('val', VALIDATION), ('test', TEST)]:
        xs, ys = [], []
        for record in records:
            path = str(raw / record)
            signal = wfdb.rdrecord(path)
            annotation = wfdb.rdann(path, 'atr')
            if signal.fs != 360:
                raise ValueError(f'{record}: expected 360 Hz, found {signal.fs}')
            if 'MLII' not in signal.sig_name:
                raise ValueError(f'{record}: missing MLII lead')
            lead = signal.p_signal[:, signal.sig_name.index('MLII')]
            x, y = windows(lead, annotation.sample, annotation.symbol)
            xs.append(x); ys.append(y)
        arrays['x_'+split] = np.concatenate(xs)
        arrays['y_'+split] = np.concatenate(ys)
        if not len(arrays['y_'+split]):
            raise ValueError(f'No beats in {split}')
        counts[split] = dict(zip(CLASSES, np.bincount(arrays['y_'+split], minlength=4).tolist()))
    arrays['train_records'] = np.array(TRAIN)
    arrays['val_records'] = np.array(VALIDATION)
    arrays['test_records'] = np.array(TEST)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

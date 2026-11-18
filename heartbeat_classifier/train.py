"""Train matched unweighted/weighted CNNs and report class-specific recall."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np


def metrics(truth, predicted):
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    support = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(4), where=support > 0)
    precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(4), where=matrix.sum(0) > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(4), where=precision+recall > 0)
    return dict(accuracy=float(matrix.trace()/max(1, matrix.sum())),
                macro_f1=float(f1.mean()), macro_recall=float(recall.mean()),
                classes={name: dict(recall=float(r) if n else None, precision=float(p), f1=float(f), support=int(n))
                         for name, r, p, f, n in zip(['N', 'S', 'V', 'F'], recall, precision, f1, support)},
                confusion_matrix=matrix.tolist())


def build_model(torch):
    nn = torch.nn
    return nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
                         nn.Flatten(), nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 4))


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='heartbeat_classifier/data/beats.npz')
    parser.add_argument('--output', default='heartbeat_classifier/artifacts')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('epochs and batch must be positive')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    # Lists avoid the NumPy bridge requirement in older PyTorch installations.
    with np.load(args.data) as data:
        tensors = {k: torch.tensor(data[k].tolist(), dtype=torch.float32 if k.startswith('x_') else torch.long)
                   for k in ('x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test')}
        groups = [set(data[k].tolist()) for k in ('train_records', 'val_records', 'test_records')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Record leakage between splits')
    if any('201' in groups[i] and '202' in groups[j] for i in range(3) for j in range(3) if i != j):
        raise ValueError('Patient leakage: 201 and 202 must stay together')
    counts = torch.bincount(tensors['y_train'], minlength=4)
    if (counts == 0).any():
        raise ValueError('Training data must contain all four classes')
    report = {'seed': args.seed, 'epochs': args.epochs, 'train_counts': counts.tolist()}
    def predict(model, split):
        model.eval()
        with torch.no_grad():
            return np.array([p for batch in tensors['x_'+split].split(args.batch)
                             for p in model(batch.unsqueeze(1)).argmax(1).tolist()])
    for weighted in (False, True):
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensors['x_train'], tensors['y_train']),
                                             batch_size=args.batch, shuffle=True, generator=generator)
        model = build_model(torch)
        weights = counts.sum() / (4*counts.float()) if weighted else None
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best, best_state, history = -1., None, []
        for epoch in range(args.epochs):
            model.train()
            total, seen = 0., 0
            for x, y in loader:
                optimizer.zero_grad(); loss = criterion(model(x.unsqueeze(1)), y)
                loss.backward(); optimizer.step()
                total += loss.item()*len(y); seen += len(y)
            validation = metrics(np.array(tensors['y_val'].tolist()), predict(model, 'val'))
            score = validation['macro_f1']
            if score > best:
                best = score; best_state = copy.deepcopy(model.state_dict())
            history.append(dict(epoch=epoch+1, loss=total/seen, val_macro_f1=score))
            print(f'weighted={weighted} epoch={epoch+1} val_macro_f1={score:.4f}', flush=True)
        model.load_state_dict(best_state)
        name = 'weighted' if weighted else 'baseline'
        torch.save(best_state, out / (name+'.pt'))
        report[name] = dict(test=metrics(np.array(tensors['y_test'].tolist()), predict(model, 'test')),
                            history=history, class_weights=weights.tolist() if weighted else None)
    report['recall_change_percentage_points'] = {
        c: (report['weighted']['test']['classes'][c]['recall'] - report['baseline']['test']['classes'][c]['recall'])*100
        if report['weighted']['test']['classes'][c]['recall'] is not None else None for c in ['N', 'S', 'V', 'F']}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['recall_change_percentage_points'], indent=2))


if __name__ == '__main__':
    main()

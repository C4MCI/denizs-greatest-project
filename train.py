"""
Training & Evaluation Script
-----------------------------
Trains both PatchTST and LSTM, prints comparison table at the end.

Usage:
    python train.py                   # train both models
    python train.py --model transformer
    python train.py --model lstm
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

from data.generate_dataset import generate, split_and_save
from utils.dataset          import make_loaders
from models.transformer     import PatchTSTClassifier, LSTMClassifier


# ── config ───────────────────────────────────────────────────────────────────

CFG = dict(
    seq_len    = 60,
    n_channels = 4,
    batch_size = 128,
    epochs     = 40,
    lr         = 3e-4,
    patience   = 7,        # early stopping
    device     = 'cuda' if torch.cuda.is_available() else 'cpu',
    data_dir   = 'data',
    ckpt_dir   = 'checkpoints',
)

TRANSFORMER_CFG = dict(
    seq_len    = CFG['seq_len'],
    n_channels = CFG['n_channels'],
    patch_len  = 12,     # 60 / 12 = 5 patches
    d_model    = 64,
    n_heads    = 4,
    n_layers   = 3,
    d_ff       = 128,
    dropout    = 0.1,
)

LSTM_CFG = dict(
    n_channels = CFG['n_channels'],
    hidden_dim = 64,
    n_layers   = 2,
    dropout    = 0.2,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_class_weight(loader):
    """Pos-weight for BCEWithLogitsLoss from training labels."""
    ys = [y for _, y in loader.dataset]
    ys = torch.stack(ys)
    neg, pos = (ys == 0).sum().item(), (ys == 1).sum().item()
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(dim=1)
    probs  = torch.softmax(logits, dim=1)[:, 1]

    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if labels.unique().numel() > 1 else 0.0
    acc = (preds == labels).float().mean().item()
    loss = nn.CrossEntropyLoss()(logits, labels).item()
    return dict(loss=loss, acc=acc, f1=f1, auc=auc, preds=preds, labels=labels)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def train(model, name, train_loader, val_loader, cfg):
    device = cfg['device']
    model  = model.to(device)

    pos_w     = compute_class_weight(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.stack([torch.ones(1).squeeze().to(device), pos_w.squeeze()]))
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    best_f1, patience_cnt = 0.0, 0
    history = []

    print(f'\n{"─"*60}')
    print(f'  Training {name}  ({model.count_params():,} parameters)')
    print(f'{"─"*60}')
    print(f'  {"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>9}  '
          f'{"Val Acc":>8}  {"Val F1":>7}  {"Val AUC":>8}')
    print(f'  {"─"*5}  {"─"*10}  {"─"*9}  {"─"*8}  {"─"*7}  {"─"*8}')

    t0 = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        history.append(dict(epoch=epoch, tr_loss=tr_loss, **metrics))

        improved = metrics['f1'] > best_f1
        flag = ' ★' if improved else ''
        print(f'  {epoch:>5}  {tr_loss:>10.4f}  {metrics["loss"]:>9.4f}  '
              f'{metrics["acc"]:>8.4f}  {metrics["f1"]:>7.4f}  '
              f'{metrics["auc"]:>8.4f}{flag}')

        if improved:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(),
                       f'{cfg["ckpt_dir"]}/{name}_best.pt')
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg['patience']:
                print(f'\n  Early stopping at epoch {epoch}.')
                break

    elapsed = time.time() - t0
    print(f'\n  Best val F1 : {best_f1:.4f}  |  Time : {elapsed:.1f}s')
    return history


def final_test(model, name, test_loader, cfg):
    ckpt = f'{cfg["ckpt_dir"]}/{name}_best.pt'
    model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
    m = evaluate(model, test_loader, cfg['device'])
    print(f'\n{"─"*60}')
    print(f'  {name}  –  TEST RESULTS')
    print(f'{"─"*60}')
    print(f'  Accuracy : {m["acc"]:.4f}')
    print(f'  F1 Score : {m["f1"]:.4f}')
    print(f'  ROC-AUC  : {m["auc"]:.4f}')
    print('\n' + classification_report(m['labels'], m['preds'],
                                       target_names=['Normal', 'Failure']))
    cm = confusion_matrix(m['labels'], m['preds'])
    print(f'  Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n'
          f'  FN={cm[1,0]}  TP={cm[1,1]}')
    return m


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['transformer', 'lstm', 'both'],
                        default='both')
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    parser.add_argument('--regen', action='store_true',
                        help='Regenerate the dataset even if it exists')
    args = parser.parse_args()
    CFG['epochs'] = args.epochs

    # ── data ──────────────────────────────────────────────────────────────
    if args.regen or not os.path.exists('data/train_X.npy'):
        print('Generating synthetic dataset …')
        df = generate()
        split_and_save(df)

    print('\nLoading data loaders …')
    train_loader, val_loader, test_loader, _ = make_loaders(
        CFG['data_dir'], CFG['batch_size'])
    print(f'Device : {CFG["device"]}')

    results = {}

    if args.model in ('transformer', 'both'):
        model = PatchTSTClassifier(**TRANSFORMER_CFG)
        train(model, 'PatchTST', train_loader, val_loader, CFG)
        results['PatchTST'] = final_test(model, 'PatchTST', test_loader, CFG)

    if args.model in ('lstm', 'both'):
        model = LSTMClassifier(**LSTM_CFG)
        train(model, 'LSTM', train_loader, val_loader, CFG)
        results['LSTM'] = final_test(model, 'LSTM', test_loader, CFG)

    # ── comparison table ──────────────────────────────────────────────────
    if len(results) > 1:
        print(f'\n{"═"*60}')
        print(f'  FINAL COMPARISON')
        print(f'{"═"*60}')
        print(f'  {"Model":>12}  {"Accuracy":>9}  {"F1":>7}  {"ROC-AUC":>8}')
        print(f'  {"─"*12}  {"─"*9}  {"─"*7}  {"─"*8}')
        for name, m in results.items():
            print(f'  {name:>12}  {m["acc"]:>9.4f}  {m["f1"]:>7.4f}  {m["auc"]:>8.4f}')
        print(f'{"═"*60}')


if __name__ == '__main__':
    main()

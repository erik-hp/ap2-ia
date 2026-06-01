"""Ponto de entrada para treinar e registrar experimentos do PathMNIST."""

from __future__ import annotations

import argparse

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from data.dataset import get_loaders
from models.custom_cnn import CustomCNN
from models.transfer import create_model, parameter_groups
from utils import EarlyStopping, Timer, append_result, device, set_seed


def run_epoch(model, loader, criterion, optimizer=None, current_device=None):
    # Quando optimizer e None, a funcao roda em modo avaliacao.
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, leave=False):
        labels = labels.squeeze().long()
        images, labels = images.to(current_device), labels.to(current_device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                # set_to_none=True reduz uso de memoria e evita zerar tensores a mao.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred)


def build_model(name: str, mode: str):
    # A CNN autoral nao usa modos de transfer learning.
    if name == "custom_cnn":
        return CustomCNN()
    return create_model(name, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="custom_cnn", choices=["custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "vit_b_16"])
    parser.add_argument("--mode", default="feature_extraction", choices=["feature_extraction", "fine_tuning"])
    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results", default="experiments/results.csv")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    current_device = device()
    loaders = get_loaders(batch_size=args.batch_size)
    model = build_model(args.model, args.mode).to(current_device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    groups = parameter_groups(model, args.lr, args.mode) if args.model != "custom_cnn" else model.parameters()
    optimizer = SGD(groups, lr=args.lr, momentum=0.9) if args.optimizer == "sgd" else AdamW(groups, lr=args.lr, weight_decay=1e-4)

    # Opcoes exigidas para a Etapa 5: cosine annealing, label smoothing e early stopping.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None
    stopper = EarlyStopping(patience=args.patience, mode="min") if args.early_stopping else None

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with Timer() as timer:
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, current_device)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, current_device)
            if scheduler is not None:
                scheduler.step()

        vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        append_result(args.results, {
            "modelo": args.model,
            "modo": args.mode,
            "otimizador": args.optimizer,
            "lr": args.lr,
            "epoca": epoch,
            "loss_train": train_loss,
            "loss_val": val_loss,
            "acc_train": train_acc,
            "acc_val": val_acc,
            "tempo_s": timer.elapsed,
            "vram_mb": vram_mb,
        })
        if stopper is not None and stopper.step(val_loss):
            break


if __name__ == "__main__":
    main()

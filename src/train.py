"""Ponto de entrada para treinar e registrar experimentos do PathMNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from data.dataset import PATHMNIST_CLASSES, get_loaders
from evaluate import predict_loader, save_classification_artifacts
from models.custom_cnn import CustomCNN
from models.transfer import MODEL_NAMES, create_model, parameter_groups
from utils import EarlyStopping, Timer, append_result, collect_hardware_info, device, save_json, set_seed


def run_epoch(model, loader, criterion, optimizer=None, current_device=None):
    """Executa uma epoca de treino ou avaliacao.

    :param model: Modelo PyTorch.
    :param loader: DataLoader da epoca.
    :param criterion: Funcao de perda.
    :param optimizer: Otimizador; ``None`` ativa avaliacao.
    :param current_device: Dispositivo; quando ausente, e detectado.
    :return: Tupla ``(loss_media, accuracy)``.
    :raises ValueError: Se o loader nao produzir amostras.
    """
    if current_device is None:
        current_device = device()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in tqdm(loader, leave=False):
        labels = labels.view(-1).long()
        images = images.to(current_device, non_blocking=True)
        labels = labels.to(current_device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                # set_to_none=True reduz uso de memoria e evita zerar tensores a mao.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.numel()

    if total_items == 0:
        raise ValueError("loader produced no samples")
    return total_loss / total_items, total_correct / total_items


def build_model(name: str, mode: str, pretrained: bool = True):
    """Cria a CNN autoral ou um backbone torchvision."""
    if name == "custom_cnn":
        return CustomCNN()
    return create_model(name, mode=mode, pretrained=pretrained)


def _is_improved(value: float, best: float | None, mode: str) -> bool:
    """Retorna se uma metrica melhorou segundo ``min`` ou ``max``."""
    return best is None or (value < best if mode == "min" else value > best)


def _wandb_init(args):
    """Inicializa WandB somente quando explicitamente solicitado."""
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WandB is not installed. Run: pip install wandb") from exc
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.run_name or None, config=vars(args))


def main():
    """Executa treino, logging CSV/JSON, checkpoint e rastreamento opcional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="custom_cnn", choices=["custom_cnn", *MODEL_NAMES])
    parser.add_argument("--mode", default="feature_extraction", choices=["feature_extraction", "fine_tuning"])
    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--source-size", type=int, default=224, choices=[28, 64, 128, 224])
    parser.add_argument("--augment-policy", default="basic", choices=["none", "basic", "randaugment", "autoaugment"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--results", default="experiments/results.csv")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-metric", default="val_loss", choices=["val_loss", "val_acc"])
    parser.add_argument("--checkpoint-mode", default="auto", choices=["auto", "min", "max"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-metadata", default="")
    parser.add_argument("--evaluation-dir", default="")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ap2-ia-pathmnist")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.source_size != 224:
        raise ValueError("Official project rules require --source-size 224 for PyTorch stages.")

    set_seed(args.seed)
    current_device = device()
    wandb_run = _wandb_init(args)
    if args.run_metadata:
        save_json(
            args.run_metadata,
            {
                "args": vars(args),
                "hardware": collect_hardware_info(),
            },
        )
    loaders = get_loaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        source_size=args.source_size,
        num_workers=args.num_workers,
        augment_policy=args.augment_policy,
    )
    model = build_model(args.model, args.mode, pretrained=not args.no_pretrained).to(current_device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    groups = parameter_groups(model, args.lr, args.mode) if args.model != "custom_cnn" else model.parameters()
    optimizer = SGD(groups, lr=args.lr, momentum=0.9, nesterov=True) if args.optimizer == "sgd" else AdamW(groups, lr=args.lr, weight_decay=1e-4)

    # Opcoes exigidas para a Etapa 5: cosine annealing, label smoothing e early stopping.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None
    stopper = EarlyStopping(patience=args.patience, mode="min") if args.early_stopping else None
    checkpoint_mode = args.checkpoint_mode
    if checkpoint_mode == "auto":
        checkpoint_mode = "min" if args.checkpoint_metric == "val_loss" else "max"
    best_checkpoint_value: float | None = None

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with Timer() as timer:
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, current_device)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, current_device)
            if scheduler is not None:
                scheduler.step()

        vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        epoch_metrics = {
            "tag": args.tag or f"{args.model}_{args.mode}_{args.optimizer}_{args.lr}",
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
        }
        append_result(args.results, epoch_metrics)
        if wandb_run is not None:
            wandb_run.log({**epoch_metrics, "epoch": epoch})

        checkpoint_value = val_loss if args.checkpoint_metric == "val_loss" else val_acc
        if args.checkpoint and _is_improved(checkpoint_value, best_checkpoint_value, checkpoint_mode):
            best_checkpoint_value = checkpoint_value
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": args.model,
                    "mode": args.mode,
                    "num_classes": 9,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "checkpoint_metric": args.checkpoint_metric,
                    "checkpoint_mode": checkpoint_mode,
                },
                checkpoint_path,
            )
            if wandb_run is not None:
                import wandb

                artifact = wandb.Artifact(f"{wandb_run.id}-best-model", type="model")
                artifact.add_file(str(checkpoint_path))
                wandb_run.log_artifact(artifact)
        if stopper is not None and stopper.step(val_loss):
            break

    if args.evaluation_dir:
        if args.checkpoint and Path(args.checkpoint).exists():
            checkpoint = torch.load(args.checkpoint, map_location=current_device)
            model.load_state_dict(checkpoint["model_state_dict"])
        y_true, y_pred = predict_loader(model, loaders["val"], current_device)
        save_classification_artifacts(y_true, y_pred, PATHMNIST_CLASSES, args.evaluation_dir, prefix="val")
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(f"{wandb_run.id}-validation-report", type="evaluation")
            for path in Path(args.evaluation_dir).glob("val_*"):
                artifact.add_file(str(path))
            wandb_run.log_artifact(artifact)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

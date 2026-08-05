"""The training loop."""

import datetime
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config
from evaluation import evaluate_model

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, class_weights=None,
                epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                num_classes=config.NUM_CLASSES,
                checkpoint_dir=config.REPO_ROOT / "checkpoints",
                run=None, device=None):
    """Train the model, keeping the weights that score best on validation.

    Only the best epoch is written to disk, so the checkpoint left behind is the
    one that generalised best rather than whatever the last epoch produced.

    Pass a Weights & Biases run as `run` to log metrics; leave it None and wandb
    is never imported.

    Returns a dict with the per-epoch losses and accuracies, the final confusion
    matrix, the best validation accuracy and the checkpoint path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    model.to(device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"cnn_checkpoint_{timestamp}.pth"

    if class_weights is None:
        loss_function = nn.CrossEntropyLoss(ignore_index=config.PADDING_LABEL)
    else:
        loss_function = nn.CrossEntropyLoss(
            weight=class_weights.float().to(device),
            ignore_index=config.PADDING_LABEL)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    best_accuracy = 0.0
    metrics = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for features, labels in train_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            # Flatten batch and depth together: every bin is its own sample.
            logits = model(features).reshape(-1, num_classes)
            loss = loss_function(logits, labels.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses[epoch] = epoch_loss / len(train_loader)

        metrics = evaluate_model(model, val_loader, loss_function, num_classes)
        val_losses[epoch] = metrics["loss"]
        val_accuracies[epoch] = metrics["accuracy"]

        logger.info("Epoch %d/%d | train loss %.6f | val loss %.6f | val accuracy %.4f",
                    epoch + 1, epochs, train_losses[epoch], val_losses[epoch],
                    val_accuracies[epoch])

        if run is not None:
            run.log({"train_loss": train_losses[epoch],
                     "val_loss": val_losses[epoch],
                     "val_acc": val_accuracies[epoch]}, step=epoch)

        if metrics["accuracy"] > best_accuracy:
            logger.info("Validation accuracy %.4f -> %.4f, saving checkpoint",
                        best_accuracy, metrics["accuracy"])
            best_accuracy = metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    logger.info("Done. Best validation accuracy %.4f", best_accuracy)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "confusion_matrix": metrics.get("confusion_matrix"),
        "best_val_accuracy": best_accuracy,
        "checkpoint_path": checkpoint_path,
    }

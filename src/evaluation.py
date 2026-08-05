"""Metrics for per-depth-bin classification."""

import numpy as np
import torch
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

import config


def evaluate_model(model, dataloader, loss_function, num_classes=config.NUM_CLASSES):
    """Run the model over a dataloader and return loss plus classification metrics.

    Precision, recall and F1 are macro-averaged so that Wind Slab, which covers
    far fewer bins than the other classes, still counts for a quarter of the
    score. Padded bins are excluded throughout.

    Returns a dict with loss, accuracy, precision, recall, f1 and
    confusion_matrix.
    """
    device = next(model.parameters()).device
    model.eval()

    metric_args = dict(task="multiclass", num_classes=num_classes,
                       ignore_index=config.PADDING_LABEL)
    accuracy = Accuracy(**metric_args).to(device)
    precision = Precision(average="macro", **metric_args).to(device)
    recall = Recall(average="macro", **metric_args).to(device)
    f1 = F1Score(average="macro", **metric_args).to(device)
    confusion = ConfusionMatrix(**metric_args).to(device)

    total_loss = 0.0
    predictions, targets = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            # Flatten batch and depth together: every bin is its own sample.
            logits = model(features).reshape(-1, num_classes)
            labels = labels.reshape(-1)

            total_loss += loss_function(logits, labels).item()
            predictions.append(torch.argmax(logits, dim=1))
            targets.append(labels)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy(predictions, targets).item(),
        "precision": precision(predictions, targets).item(),
        "recall": recall(predictions, targets).item(),
        "f1": f1(predictions, targets).item(),
        "confusion_matrix": confusion(predictions, targets).cpu().numpy(),
    }


def normalize_confusion_matrix(confusion_matrix):
    """Scale each row to sum to one, so cells read as "of the true X, how many were Y"."""
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    normalized = np.array(confusion_matrix, dtype=np.float32)
    row_totals = normalized.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1  # a class with no samples stays all zeros
    return normalized / row_totals

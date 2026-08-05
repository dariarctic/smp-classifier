"""Figures for training diagnostics and profile predictions."""

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluation import normalize_confusion_matrix


def class_colormap(num_classes=config.NUM_CLASSES):
    """Sequential colours for the grain classes, skipping the palest end."""
    colours = mpl.colormaps["GnBu"](np.linspace(0, 1, num_classes + 2))
    return mpl.colors.ListedColormap(colours[-num_classes:])


def plot_confusion_matrix(confusion_matrix, class_names=config.CLASS_NAMES, ax=None):
    """Draw a row-normalized confusion matrix with the values written in."""
    normalized = normalize_confusion_matrix(confusion_matrix)
    num_classes = normalized.shape[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(normalized, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax, label="Prediction frequency")
    ax.set_title("Normalized confusion matrix (per true label)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes), class_names, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            if normalized[i, j] > 0:
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                        color="white" if normalized[i, j] > 0.5 else "black")

    ax.figure.tight_layout()
    return ax


def plot_training_history(history, ax=None):
    """Plot losses and validation accuracy against epoch."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    epochs = np.arange(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], label="Training loss")
    ax.plot(epochs, history["val_losses"], label="Validation loss")
    ax.plot(epochs, history["val_accuracies"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / accuracy")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_profile_predictions(features, predictions, true_labels=None,
                             feature_name="mean_force",
                             class_names=config.CLASS_NAMES, figsize=(5, 7)):
    """Show one profile's force trace beside its predicted grain classes.

    With true_labels given, the true and predicted columns sit side by side so
    disagreements are easy to spot.
    """
    num_classes = len(class_names)
    colormap = class_colormap(num_classes)

    values = np.asarray(features[:, config.FEATURE_VARS.index(feature_name)])
    depth = np.arange(len(values))

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(1, 2, width_ratios=[4.5, 1.4], wspace=0.15)
    ax_trace = figure.add_subplot(grid[0, 0])
    ax_classes = figure.add_subplot(grid[0, 1], sharey=ax_trace)

    ax_trace.plot(values, depth, color="black")
    ax_trace.set_xlabel(f"{feature_name.replace('_', ' ')} (normalized)")
    ax_trace.set_ylabel("Depth (bin index)")
    ax_trace.set_ylim(depth.max(), depth.min())
    ax_trace.spines[["top", "right"]].set_visible(False)
    ax_trace.legend(
        handles=[mpatches.Patch(color=colormap(i), label=f"{i} - {class_names[i]}")
                 for i in range(num_classes)],
        title="Class", loc="upper right", fontsize=9, title_fontsize=9)

    style = dict(aspect="auto", cmap=colormap, vmin=0, vmax=num_classes - 1,
                 interpolation="nearest")
    if true_labels is None:
        ax_classes.imshow(predictions.reshape(-1, 1),
                          extent=[0, 1, depth.max(), depth.min()], **style)
        ax_classes.set_xlim(0, 1)
        ax_classes.set_xticks([0.5], ["Predicted"])
    else:
        ax_classes.imshow(np.asarray(true_labels).reshape(-1, 1),
                          extent=[0, 0.95, depth.max(), depth.min()], **style)
        ax_classes.imshow(predictions.reshape(-1, 1),
                          extent=[1.05, 2, depth.max(), depth.min()], **style)
        ax_classes.set_xlim(0, 2)
        ax_classes.set_xticks([0.475, 1.525], ["True", "Predicted"])

    ax_classes.tick_params(axis="y", labelleft=False)
    ax_classes.grid(False)
    return figure

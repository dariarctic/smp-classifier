"""The grain-type classification network."""

from torch import nn

import config


class CNN1D(nn.Module):
    """Labels every depth bin of an SMP profile with a grain type.

    Three convolution blocks widen the view along depth while keeping the
    sequence length fixed, then a 1x1 convolution classifies each bin from its
    local context. Nothing pools along depth, so a profile of any length comes
    back with exactly one prediction per bin.

    The submodules have to stay named cnn and classifier: that is how the
    published checkpoints store their weights.
    """

    def __init__(self, num_classes=config.NUM_CLASSES,
                 in_channels=config.NUM_FEATURES,
                 kernel_size=config.KERNEL_SIZE,
                 dropout_rate=config.DROPOUT_RATE,
                 num_groups=1):
        super().__init__()
        padding = kernel_size // 2  # keeps the sequence length unchanged

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size, padding=padding),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 32),
            nn.Conv1d(32, 64, kernel_size, padding=padding),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 64),
            nn.Conv1d(64, 128, kernel_size, padding=padding),
            nn.ReLU(),
            nn.GroupNorm(num_groups, 128),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        """(batch, features, depth) in, (batch, depth, classes) logits out."""
        logits = self.classifier(self.cnn(x))
        return logits.transpose(1, 2)

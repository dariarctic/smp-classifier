"""Settings shared by training, evaluation and inference.

Anything that has to match between training a model and using it later lives
here, so there is one place to look when a run does not reproduce.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELLED_DATA = REPO_ROOT / "data" / "labelled" / "smp_profiles_labelled.nc"
UNLABELLED_DATA = REPO_ROOT / "data" / "unlabelled" / "PS111_profiles.nc"
CHECKPOINT = REPO_ROOT / "models" / "cnn_grain_classifier.pth"
NORM_CONSTANTS = REPO_ROOT / "models" / "normalization_constants.pkl"

# Input channels, in the order the model expects. Reordering this silently
# invalidates every existing checkpoint.
FEATURE_VARS = [
    "mean_force", "var_force", "min_force", "max_force", "distance",
    "first_derivative", "second_derivative",
    "first_absolute_derivative", "second_absolute_derivative",
    "mean_force_rolled", "var_force_rolled", "min_force_rolled", "max_force_rolled",
    "force_median", "lambda", "f0", "delta", "L",
]

# The labelled dataset stores five classes: wind slab, fragmented & rounded,
# faceted, depth hoar, melt forms. The middle two cannot be told apart from the
# force signal alone, so we merge them. Index = five-class label, value = four.
MERGE_TO_FOUR = np.array([0, 1, 1, 2, 3])

CLASS_NAMES = ["Wind Slab", "Rounded & Faceted", "Depth Hoar", "Melt Forms"]

NUM_FEATURES = len(FEATURE_VARS)
NUM_CLASSES = len(CLASS_NAMES)

# Marks padded depth bins so the loss and the metrics skip them.
PADDING_LABEL = -100

# Hyper-parameters of the published model. The kernel size is the one that
# matters: 21 depth bins, wide enough to see a whole layer at once, and what
# the shipped checkpoint expects.
KERNEL_SIZE = 21
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.005345
WEIGHT_DECAY = 0.000567
BATCH_SIZE = 32
NUM_EPOCHS = 25
SPLIT_SEED = 42

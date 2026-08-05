"""Loading SMP profiles and preparing them for the network.

One profile is a sequence of depth bins. The model labels every bin, so a
training sample is a whole profile rather than a single bin.
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config


def merge_to_four_classes(dataset, label_var="dominant_label"):
    """Merge the five stored labels into the four classes the model predicts.

    Unlabelled bins stay NaN.
    """
    labels = dataset[label_var].values
    merged = np.full_like(labels, np.nan, dtype=float)
    labelled = np.isfinite(labels)
    merged[labelled] = config.MERGE_TO_FOUR[labels[labelled].astype(int)]
    dataset[label_var] = (dataset[label_var].dims, merged)
    return dataset


def split_profiles(dataset, val_fraction=0.2, test_fraction=0.0,
                   seed=config.SPLIT_SEED):
    """Split whole profiles into training, validation and test sets.

    Splitting profiles rather than individual depth bins matters: neighbouring
    bins are highly correlated, so mixing them across splits would flatter the
    scores. Seeded, so the split is the same everywhere.

    With test_fraction=0 the returned test array is empty, which is how the
    published model was trained.
    """
    profile_ids = np.unique(dataset.profile.values)
    np.random.RandomState(seed).shuffle(profile_ids)

    n = len(profile_ids)
    n_val = int(val_fraction * n)
    n_test = int(test_fraction * n)
    n_train = n - n_val - n_test

    return (profile_ids[:n_train],
            profile_ids[n_train:n_train + n_val],
            profile_ids[n_train + n_val:])


def compute_normalization_constants(dataset, profile_ids):
    """Per-feature mean and standard deviation from the training profiles only.

    Using the whole dataset here would leak validation and test statistics into
    training.
    """
    training_profiles = dataset.sel(profile=list(profile_ids))

    constants = {}
    for var in config.FEATURE_VARS:
        values = training_profiles[var].values.flatten()
        values = values[~np.isnan(values)]
        constants[var + "_mean"] = values.mean()
        constants[var + "_std"] = values.std()
    return constants


def compute_class_weights(dataset, profile_ids, label_var="dominant_label"):
    """Inverse-frequency weights, so the loss is not dominated by one class.

    Rounded & Faceted covers about a third of all bins; without weighting the
    model can score well by simply over-predicting it.
    """
    labels = dataset.sel(profile=list(profile_ids))[label_var].values.flatten()
    labels = labels[~np.isnan(labels)]
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return torch.from_numpy(weights)


class SMPProfileDataset(Dataset):
    """Serves one complete SMP profile per item.

    Each profile is z-scored with the given constants and stripped of depth bins
    where any feature is NaN. A profile left with nothing is dropped and its id
    recorded in skipped_profile_ids, so profile_ids always lines up with what
    the dataset actually returns.

    normalization_constants is required on purpose: falling back to per-profile
    statistics would make a prediction depend on which profiles it was loaded
    alongside.
    """

    def __init__(self, dataset, profile_ids, normalization_constants,
                 include_labels=True):
        self.dataset = dataset
        self.norm = normalization_constants
        self.include_labels = include_labels

        self.profile_ids = []
        self.profiles = []
        self.skipped_profile_ids = []

        for profile_id in profile_ids:
            profile = self._prepare(profile_id)
            if profile is None:
                self.skipped_profile_ids.append(profile_id)
            else:
                self.profile_ids.append(profile_id)
                self.profiles.append(profile)

    def _prepare(self, profile_id):
        """Select one profile, drop incomplete depth bins and normalize it."""
        profile = self.dataset.sel(profile=profile_id)

        features = profile[config.FEATURE_VARS].to_array().values
        complete_bins = ~np.any(np.isnan(features), axis=0)
        profile = profile.isel(depth_bins=complete_bins)

        if profile.sizes["depth_bins"] == 0:
            return None

        for var in config.FEATURE_VARS:
            mean = self.norm[var + "_mean"]
            std = self.norm[var + "_std"]
            profile[var] = (profile[var] - mean) / std
        return profile

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        """Return (features, labels) of shape (n_bins, n_features) and (n_bins,).

        labels is None when the dataset was built without them.
        """
        profile = self.profiles[idx]
        features = np.stack(
            [profile[var].values for var in config.FEATURE_VARS], axis=1)
        features = torch.tensor(features, dtype=torch.float32)

        if not self.include_labels:
            return features, None

        labels = profile["dominant_label"].values.astype(np.int64)
        return features, torch.tensor(labels, dtype=torch.long)


def pad_and_collate(batch):
    """Pad a batch of labelled profiles to equal length, for the DataLoader.

    Label padding uses config.PADDING_LABEL so the loss and metrics ignore it.
    """
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True).permute(0, 2, 1)
    padded_labels = pad_sequence(labels, batch_first=True,
                                 padding_value=config.PADDING_LABEL)
    return padded_features, padded_labels

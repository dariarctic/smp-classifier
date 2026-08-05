"""Classifying profiles the model has not seen."""

import numpy as np
import torch
import xarray as xr

import config


def predict_profiles(model, profile_dataset, device=None):
    """Predict a grain class for every depth bin, one profile at a time.

    Profiles go through singly rather than in padded batches, so no prediction
    can be influenced by padding from a neighbouring profile.

    Returns one integer array per profile, ordered like
    profile_dataset.profile_ids.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    predictions = []
    with torch.no_grad():
        for features, _ in profile_dataset:
            # (depth, features) -> (1, features, depth)
            logits = model(features.unsqueeze(0).permute(0, 2, 1).to(device))
            logits = logits.reshape(-1, logits.shape[-1])
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return predictions


def add_predictions_to_dataset(dataset, predictions, profile_ids,
                               variable_name="predicted_labels"):
    """Attach the predictions to the dataset as a new variable.

    Written by profile id rather than by position, so a skipped profile leaves a
    row of NaN instead of shifting every later profile's results by one.
    """
    prediction_array = np.full(
        (dataset.sizes["profile"], dataset.sizes["depth_bins"]), np.nan)
    rows = {profile: i for i, profile in enumerate(dataset.profile.values)}

    for profile_id, profile_predictions in zip(profile_ids, predictions):
        prediction_array[rows[profile_id], :len(profile_predictions)] = profile_predictions

    dataset[variable_name] = xr.DataArray(
        prediction_array,
        dims=("profile", "depth_bins"),
        coords={"profile": dataset.profile, "depth_bins": dataset.depth_bins},
    )
    dataset[variable_name].attrs = {
        "long_name": "Predicted snow grain type",
        "classes": ", ".join(f"{i} = {name}"
                             for i, name in enumerate(config.CLASS_NAMES)),
    }
    return dataset

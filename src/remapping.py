"""Mapping coarse (1 mm depth-bin) predictions back onto the original,
fine-resolution SMP depth grid.
"""

import numpy as np


def remap_predictions_to_fine_resolution(fine, coarse, coarse_var="predicted_labels",
                                         out_var="predicted_labels", depth_bin_mm=1):
    """Broadcast each profile's depth-bin predictions onto its raw depth samples.

    Matches profiles between `fine` (dims profile, sample; see raw_import) and
    `coarse` (dims profile, depth_bins; see feature_engineering) by the `name`
    coordinate, then assigns every fine-resolution sample the label of the
    1 mm bin it falls into. Profiles present in `fine` but not in `coarse`,
    and samples past the end of a profile's coarse predictions, are left NaN.
    """
    coarse_names = np.asarray(coarse["name"].values, dtype=str)
    name_to_index = {name: i for i, name in enumerate(coarse_names)}

    n_profiles = fine.sizes["profile"]
    n_samples = fine.sizes["sample"]
    predicted = np.full((n_profiles, n_samples), np.nan)

    for i in range(n_profiles):
        name = str(fine["name"].isel(profile=i).item())
        j = name_to_index.get(name)
        if j is None:
            continue

        labels = coarse[coarse_var].isel(profile=j).values
        depth = fine["depth"].isel(profile=i).values

        bin_index = np.floor(depth / depth_bin_mm)
        in_range = np.isfinite(bin_index) & (bin_index >= 0) & (bin_index < len(labels))
        predicted[i, in_range] = labels[bin_index[in_range].astype(int)]

    out = fine.copy()
    out[out_var] = (("profile", "sample"), predicted)
    if coarse_var in coarse.variables:
        out[out_var].attrs = coarse[coarse_var].attrs
    return out

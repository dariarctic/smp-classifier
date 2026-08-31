"""Aggregating fine-resolution SMP profiles into the depth-bin feature
representation the CNN is trained on.

Each profile is grouped into 1 mm depth bins and reduced to the 18
force-derived features in config.FEATURE_VARS, plus a dominant grain-type
label per bin for profiles that carry one.

The depth-bin aggregation and shot-noise feature calculation are adapted from
Julia Kaltenborn's snowdragon (https://github.com/liellnima/snowdragon,
MIT License), credited again at `_poisson_params`.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mode
from snowmicropyn import loewe2012, windowing

import config
from raw_import import LABEL_CODES

# LABEL_CODES grain-type integers -> the 5-class scheme config.MERGE_TO_FOUR
# expects as input (confirmed against two independent messy-code sources: the
# commented-out remap in an earlier "Preprocessor" class, and a standalone
# filter/remap notebook). Structural markers (surface, ground, unknown) and
# grain types the CNN doesn't cover (new snow, ice forms, surface hoar) map to
# NaN and are excluded from the per-bin label vote, not from the feature
# stats.
RAW_LABEL_TO_GRAIN_CLASS = {
    LABEL_CODES["ws"]: 0,  # wind slab
    LABEL_CODES["df"]: 1,  # fragmented & rounded
    LABEL_CODES["fc"]: 2,  # faceted
    LABEL_CODES["dh"]: 3,  # depth hoar
    LABEL_CODES["mf"]: 4,  # melt forms
}

POISSON_VARS = ["force_median", "lambda", "f0", "delta", "L"]


def _remap_labels(label):
    remapped = np.full(label.shape, np.nan)
    for raw_value, grain_class in RAW_LABEL_TO_GRAIN_CLASS.items():
        remapped[label == raw_value] = grain_class
    return remapped


def _aggregate_bin(bin_samples):
    """Reduce one profile's samples in one depth bin to a single row of features.

    `dominant_label` is always computed, even for unlabelled profiles (where
    it comes out NaN in every bin) — this matches the convention of the
    published `PS111_profiles.nc`, which carries the same all-NaN variable so
    that labelled and unlabelled feature files share one schema.
    """
    force = bin_samples["force"].values
    first = np.diff(force)
    second = np.diff(first)
    dominant = mode(_remap_labels(bin_samples["label"].values),
                    nan_policy="omit", keepdims=False).mode

    # A bin with 0 or 1 samples (a handful of near-empty profiles) has no
    # differences to average; nanmean of an empty slice is legitimately NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        first_derivative = np.nanmean(first)
        second_derivative = np.nanmean(second)
        first_absolute_derivative = np.nanmean(np.abs(first))
        second_absolute_derivative = np.nanmean(np.abs(second))

    return xr.Dataset({
        "mean_force": bin_samples["force"].mean(),
        "var_force": bin_samples["force"].var(),
        "min_force": bin_samples["force"].min(),
        "max_force": bin_samples["force"].max(),
        "distance": bin_samples["depth"].max(),
        "first_derivative": (["depth_bins"], [first_derivative]),
        "second_derivative": (["depth_bins"], [second_derivative]),
        "first_absolute_derivative": (["depth_bins"], [first_absolute_derivative]),
        "second_absolute_derivative": (["depth_bins"], [second_absolute_derivative]),
        "dominant_label": (["depth_bins"], [dominant]),
    })


def _rolling_stats(binned, window):
    """Rolling mean, clamped to the profile's own bin count.

    A handful of profiles are only a few bins long (a near-empty SMP
    measurement); the rolling window can't exceed that.
    """
    window = min(window, binned.sizes["depth_bins"])
    for var in ("mean_force", "var_force", "min_force", "max_force"):
        binned[f"{var}_rolled"] = binned[var].rolling(
            depth_bins=window, center=True, min_periods=1).mean()
    return binned


def _poisson_params(profile, window):
    """Löwe et al. (2012) shot-noise parameters, one row per `window`-mm chunk.

    `window` doubles as both the rolling-window width (in depth bins) used
    for `_rolling_stats` and the chunk width (in mm) here: with 1 mm depth
    bins the two coincide, so a `window`-bin rolling average and a
    `window`-mm shot-noise chunk cover the same span.

    Adapted from snowdragon's data_handling.data_preprocessing.calc
    (https://github.com/liellnima/snowdragon, MIT License), itself credited
    there to Henning Löwe.
    """
    depth = profile["depth"].values
    valid = np.isfinite(depth)
    depth = depth[valid]
    force = pd.Series(profile["force"].values[valid]).fillna(0)

    spatial_res = np.median(np.diff(depth))
    overlap = ((window - 1) / window) * 100 + 1e-4
    chunks = windowing.chunkup(pd.DataFrame({"distance": depth, "force": force}),
                               window, overlap)

    rows = []
    for center, chunk in chunks:
        chunk_force = chunk["force"]
        step = (0.0, 0.0, 0.0, 0.0) if (chunk_force == 0).all() \
            else loewe2012.calc_step(spatial_res, chunk_force)
        rows.append((center, np.median(chunk_force), *step))

    return pd.DataFrame(rows, columns=["distance"] + POISSON_VARS)


def depth_bin_features(profile, depth_bin_mm=1, rolling_window=4):
    """Aggregate one fine-resolution profile into depth_bins-indexed features.

    Produces every variable in config.FEATURE_VARS, plus dominant_label.
    """
    bin_id = profile["depth"] // depth_bin_mm
    binned = profile.assign_coords(depth_bins=("sample", bin_id.values)).dropna(
        "sample", subset=["depth_bins"])
    binned["depth_bins"] = binned["depth_bins"].astype(int)

    grouped = binned.groupby("depth_bins").map(_aggregate_bin)
    grouped = grouped.assign_coords(depth_bins=grouped["depth_bins"].values)

    grouped = _rolling_stats(grouped, rolling_window)

    poisson = _poisson_params(profile, rolling_window)
    n_bins = grouped.sizes["depth_bins"]
    for var in POISSON_VARS:
        values = poisson[var].to_numpy()[:n_bins]
        if len(values) < n_bins:
            values = np.pad(values, (0, n_bins - len(values)), constant_values=np.nan)
        grouped[var] = ("depth_bins", values)

    return grouped[config.FEATURE_VARS + ["dominant_label"]]


def build_feature_dataset(campaign, depth_bin_mm=1, rolling_window=4):
    """Aggregate every profile in a fine-resolution campaign Dataset.

    Keeps the `name`/`time`/`lat`/`lon`/`icetype` coordinates that
    `02_inference.ipynb` and `remapping.remap_predictions_to_fine_resolution`
    rely on.
    """
    profiles = []
    for i in range(campaign.sizes["profile"]):
        profile = campaign.isel(profile=i)
        features = depth_bin_features(profile, depth_bin_mm, rolling_window)
        features = features.assign_coords(
            name=profile["name"], time=profile["time"],
            lat=profile["lat"], lon=profile["lon"], icetype=profile["icetype"])
        profiles.append(features)

    dataset = xr.concat(profiles, dim="profile", join="outer")
    dataset["depth_bins"].attrs = dict(
        long_name="snow depth", units="mm",
        description="snow depth as distance from the air-snow interface")
    return _add_attrs(dataset)


def _add_attrs(ds):
    attrs = {
        "mean_force": dict(long_name="mean snow resistance force within the depth bin",
                           units="N"),
        "var_force": dict(long_name="variance of force within the depth bin", units="N^2"),
        "min_force": dict(long_name="minimum force within the depth bin", units="N"),
        "max_force": dict(long_name="maximum force within the depth bin", units="N"),
        "distance": dict(long_name="distance to the snow surface", units="mm",
                         description="measured at the top of the depth bin"),
        "first_derivative": dict(
            long_name="first derivative of the force within the depth bin", units="N/mm"),
        "second_derivative": dict(
            long_name="second derivative of the force within the depth bin", units="N/mm^2"),
        "first_absolute_derivative": dict(
            long_name="absolute first derivative of the force within the depth bin",
            units="N/mm"),
        "second_absolute_derivative": dict(
            long_name="absolute second derivative of the force within the depth bin",
            units="N/mm^2"),
        "mean_force_rolled": dict(
            long_name="mean force after the rolling window", units="N"),
        "var_force_rolled": dict(
            long_name="variance of force after the rolling window", units="N^2"),
        "min_force_rolled": dict(
            long_name="minimum force after the rolling window", units="N"),
        "max_force_rolled": dict(
            long_name="maximum force after the rolling window", units="N"),
        "force_median": dict(long_name="median force in the shot-noise window", units="N"),
        "lambda": dict(long_name="lambda from the Löwe et al. (2012) model, "
                                 "intensity of the point process", units="1/mm"),
        "f0": dict(long_name="f0 from the Löwe et al. (2012) model, mean rupture force",
                  units="N"),
        "delta": dict(long_name="delta from the Löwe et al. (2012) model, "
                                "deflection at rupture", units="mm"),
        "L": dict(long_name="L from the Löwe et al. (2012) model, element size", units="mm"),
        "dominant_label": dict(
            long_name="dominant snow grain type in the depth bin",
            description="0 = wind slab, 1 = fragmented & rounded, 2 = faceted, "
                        "3 = depth hoar, 4 = melt forms"),
    }
    for var, var_attrs in attrs.items():
        if var in ds.variables:
            ds[var].attrs = var_attrs
    return ds

"""Building a fine-resolution NetCDF for one SMP campaign from raw field data.

A campaign directory holds one folder per SMP station (e.g.
``PS111_20180211_3-2_SMP``), each containing one or more profile groups —
usually a walked transect (``*TRANS*``), sometimes also one or more
snow-pit-side groups (``*SPIT*``). Each group folder has a tab-separated
``*_ini.txt`` table identifying each profile (by a bare file number or by its
full name, depending on the campaign) alongside its location and manually
measured snow depth, and a ``_raw`` (or ``raw``) subfolder of ``.pnt`` files.
Every SMP
station has a matching snowpit folder under a separate root, whose name
shares the station's date and number (``PS111_20180211_3-2_SMP`` <->
``PS111_20180211_3-1_SPIT0N``) and holds a ``*META.txt`` with position, ice
thickness, freeboard and ice type.

Each profile keeps its own sample count: profiles are concatenated along a
plain ``profile`` dimension without aligning the ``sample`` dimension by
depth value, so the result stays one array per profile (padded with NaN to
the longest profile) instead of the enormous, mostly-empty array that a
value-aligned concat over slightly different depth grids would produce.

Besides force and label, each profile also carries King et al. (2020b) and
Calonne & Richter (2020) density/SSA retrievals, so the fine-resolution file
is useful on its own before (or instead of) running it through the CNN. Those
retrievals come back on their own, coarser regular grids (not the original
sample positions), so they are placed onto the nearest fine-resolution sample
and left NaN elsewhere, the same way the original notebook this is based on
merged them by depth.

The force-signal cleanup and layer-marker labelling below are adapted from
Julia Kaltenborn's snowdragon (https://github.com/liellnima/snowdragon,
MIT License), credited again at each function.
"""

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import snowmicropyn as smp
import xarray as xr
from snowmicropyn import Profile

# Grain-type / marker integers written into `label`, matching the marker
# names the SMP field software writes. surface/ground/unknown are structural
# markers, not grain types.
LABEL_CODES = {
    "unknown": 0, "surface": 1, "ground": 2, "ns": 3, "ws": 4,
    "df": 5, "fc": 6, "dh": 7, "mf": 8, "if": 9, "sh": 10,
}

_DIGITS = str.maketrans("", "", "0123456789")


def _visible(paths):
    """Drop dotfiles — mainly macOS AppleDouble ("._foo") resource forks that
    tag along whenever field data has passed through a Mac, and that
    otherwise glob-match right alongside the real file they shadow.
    """
    return sorted(p for p in paths if not Path(p).name.startswith("."))


def _label_from_markers(samples, profile):
    """Assign a grain-type label to every sample from the profile's layer markers.

    Adapted from snowdragon's data_handling.data_preprocessing.label_pd
    (https://github.com/liellnima/snowdragon, MIT License).
    """
    dist = samples.distance.to_numpy()
    force = samples.force.to_numpy()
    label = np.full_like(dist, np.nan)

    last_marker_depth = profile.markers.get("surface")
    for marker in sorted(profile.markers, key=profile.markers.get):
        if marker in ("surface", "unknown", "ground"):
            continue
        marker_depth = profile.markers[marker]
        in_layer = (dist > last_marker_depth) & (dist <= marker_depth)
        label[in_layer] = LABEL_CODES[marker.translate(_DIGITS)]
        last_marker_depth = marker_depth

    return dist, force, label


def _remove_negative_force(force, drop_threshold=-1):
    """Zero out small negative noise; replace larger dips with a local average.

    More than `drop_threshold` below zero is treated as a sensor glitch rather
    than noise, and is replaced by the mean of the nearest valid readings on
    either side (0 for a glitch with no valid reading on that side at all).

    The approach (not the implementation, which is vectorized here) is
    snowdragon's data_handling.data_preprocessing.remove_negatives
    (https://github.com/liellnima/snowdragon, MIT License).
    """
    force = np.where((force < 0) & (force > drop_threshold), 0, force)

    bad = force < drop_threshold
    good_idx = np.flatnonzero(~bad)
    if not bad.any() or good_idx.size == 0:
        return np.where(bad, 0, force)

    bad_idx = np.flatnonzero(bad)
    insert_pos = np.searchsorted(good_idx, bad_idx)
    before = good_idx[np.maximum(insert_pos - 1, 0)]
    after = good_idx[np.minimum(insert_pos, good_idx.size - 1)]

    before_value = np.where(insert_pos > 0, force[before], 0)
    after_value = np.where(insert_pos < good_idx.size, force[after], 0)
    force[bad_idx] = (before_value + after_value) / 2
    return force


def _place_on_grid(dist, sparse_distance, sparse_values):
    """Place (sparse_distance, sparse_values) pairs onto their nearest sample in `dist`.

    `dist` is sorted; every other sample not hit by a nearest match stays NaN.
    """
    grid = np.full(dist.shape, np.nan)
    if sparse_distance.size == 0:
        return grid

    right = np.clip(np.searchsorted(dist, sparse_distance), 0, len(dist) - 1)
    left = np.clip(right - 1, 0, len(dist) - 1)
    left_is_closer = np.abs(dist[left] - sparse_distance) < np.abs(dist[right] - sparse_distance)
    nearest = np.where(left_is_closer, left, right)
    grid[nearest] = sparse_values
    return grid


def _derived_parameters(dist, force):
    """King et al. (2020b) density and Calonne & Richter (2020) density/SSA.

    Both retrievals come back on their own coarser, regularly spaced grid
    (not `dist`), so their values are placed onto the nearest fine-resolution
    sample rather than assumed to line up positionally.
    """
    samples = pd.DataFrame({"distance": dist, "force": force})
    # A near-zero or negative median force in one of CR2020's windows sends
    # its density formula's log(F_m) to -inf or NaN; the warning is expected
    # and harmless — that one degenerate estimate is used as-is, same as any
    # other density/SSA value here (unused by the CNN, kept for reference).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        k2020b = smp.params["K2020b"].calc(samples)
        cr2020 = smp.params["CR2020"].calc(samples)
    return {
        "K2020b_density": _place_on_grid(
            dist, k2020b["distance"].to_numpy(), k2020b["K2020b_density"].to_numpy()),
        "CR2020_density": _place_on_grid(
            dist, cr2020["distance"].to_numpy(), cr2020["CR2020_density"].to_numpy()),
        "CR2020_ssa": _place_on_grid(
            dist, cr2020["distance"].to_numpy(), cr2020["CR2020_ssa"].to_numpy()),
    }


def _station_meta(datastation, snowpit_root):
    """Look up the snowpit META.txt for one SMP station.

    Snowpit and SMP folders for the same station share everything up to the
    leg suffix after the first "-" (SMP "..._1-2_SMP_A11" <-> snowpit
    "..._1-1_SPIT01_A11"), so that prefix is what locates the match.
    """
    prefix = datastation.split("-", 1)[0]
    matches = _visible(glob.glob(
        str(Path(snowpit_root) / f"{prefix}*" / "**" / "*META.txt"), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"no snowpit META.txt found for station {datastation!r} under {snowpit_root}")

    meta = pd.read_csv(matches[0], sep="\t", header=0, names=[
        "date", "time", "latitude", "longitude", "ice_thickness", "freeboard",
        "snow_depth", "floe_description", "weather", "smp_measurements"])
    meta[["ice_type", "floe_description"]] = meta["floe_description"].str.split(
        ",", n=1, expand=True)
    return meta.iloc[0]


def load_profile(pnt_path, datastation, snowpit_root):
    """Load one .pnt file into a (sample,)-indexed fine-resolution Dataset.

    Profiles with layer markers get `label` from those markers; unlabelled
    ones get the surface/ground auto-detected instead and `label` filled with
    NaN. `datastation` is the SMP station folder name (e.g.
    "AFIN25_20251113_1-2_SMP_A11"), used to find the matching snowpit meta.
    """
    profile = Profile.load(str(pnt_path))
    labelled = len(profile.markers) != 0

    if not labelled:
        profile.detect_ground()
        profile.detect_surface()
    samples = profile.samples_within_snowpack(relativize=False)

    if labelled:
        dist, force, label = _label_from_markers(samples, profile)
    else:
        dist = samples.distance.to_numpy()
        force = samples.force.to_numpy()
        label = np.full_like(dist, np.nan)

    dist = dist - dist[0]  # surface at 0

    if np.any(force < 0):
        if np.sum(force < 0) > 0.5 * len(force):
            raise ValueError(
                f"more than half the force values in profile {profile.name} are negative")
        force = _remove_negative_force(force)

    derived = _derived_parameters(dist, force)
    meta = _station_meta(datastation, snowpit_root)
    lat, lon = profile.coordinates if profile.coordinates is not None else (np.nan, np.nan)

    return xr.Dataset(
        coords={
            "time": np.datetime64(profile.timestamp.replace(tzinfo=None), "ns"),
            "lat": lat,
            "lon": lon,
            "icetype": meta.ice_type,
        },
        data_vars={
            "depth": ("sample", dist),
            "force": ("sample", force),
            "label": ("sample", label),
            **{name: ("sample", values) for name, values in derived.items()},
            "icethickness": pd.to_numeric(meta.ice_thickness, errors="coerce"),
            "freeboard": pd.to_numeric(meta.freeboard, errors="coerce"),
            "name": profile.name,
        },
    )


def _find_raw_dir(transect_dir):
    """Find the folder of .pnt files for one transect.

    Usually sits right inside the transect folder, but some campaigns keep
    one shared `_raw` per station instead (all of that station's transects
    draw from it), so the station folder is checked too.
    """
    for candidate_dir in (transect_dir, transect_dir.parent):
        for name in ("_raw", "raw"):
            candidate = candidate_dir / name
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        f"no '_raw' or 'raw' directory under {transect_dir} or its parent")


def _find_column(columns, keyword):
    """The first column whose header contains `keyword`, case-insensitively.

    Transect tables name their columns differently between campaigns (e.g.
    "file_no" vs. "actual filename"), so columns are matched by keyword
    rather than by position or an exact, campaign-specific name.
    """
    matches = [c for c in columns if keyword in c.lower()]
    if not matches:
        raise ValueError(f"no column matching {keyword!r} in {list(columns)}")
    return matches[0]


def _find_files(raw_dir, file_id, extension_glob):
    """Match a transect row's file identifier against files in `raw_dir`.

    `file_id` is either a bare number (the profile's numeric suffix, some
    campaigns) or an already-complete profile name (others); a purely numeric
    id is matched as a filename suffix, anything else matched exactly.
    """
    file_id = str(file_id).strip()
    if file_id.replace(".", "", 1).isdigit():
        file_id = str(int(float(file_id)))
        return _visible(raw_dir.glob(f"*{file_id}{extension_glob}"))
    return _visible(raw_dir.glob(f"{file_id}{extension_glob}"))


def _drop_duplicate_profiles(campaign):
    """Keep the first occurrence of each profile name.

    The same .pnt file sometimes ends up listed in more than one transect
    table; duplicates carry identical data, so only the first is kept.
    """
    _, first_seen = np.unique(campaign["name"].values, return_index=True)
    return campaign.isel(profile=np.sort(first_seen))


def build_campaign_dataset(campaign_dir, snowpit_root):
    """Build one (profile, sample)-indexed Dataset for every profile in a campaign.

    Walks every transect-style table (`*_ini.txt`) under `campaign_dir`,
    matches each row's file number to its raw .pnt file, and loads and tags
    each profile with its transect's snow_depth/group_id/location, plus a
    `training` flag (True when a matching per-profile .ini marker file sits
    next to it). Not just `*TRANS*` folders: some campaigns record a walked
    transect and one or more snow-pit-side profile groups (`*SPIT*`) as
    separate `*_ini.txt` tables of the same format, both meant to be walked
    the same way. A profile that turns out unusable — mostly negative force,
    or one `snowmicropyn` itself can't find a ground/surface signal in — is
    skipped and reported rather than aborting the whole campaign.
    """
    campaign_dir = Path(campaign_dir)
    transect_files = _visible(campaign_dir.rglob("*_ini.txt"))
    if not transect_files:
        raise FileNotFoundError(f"no transect '*_ini.txt' files found under {campaign_dir}")

    profiles = []
    for ini_path in transect_files:
        transect_dir = ini_path.parent
        station_dir = transect_dir.parent
        raw_dir = _find_raw_dir(transect_dir)

        # Some rows (typically "no snow, no profile taken") have a ragged
        # trailing-tab count that breaks strict tokenizing; skip just that
        # row rather than the whole table. Its file number would be empty
        # anyway, so it's dropped by the check below regardless.
        transect = pd.read_csv(ini_path, sep="\t", engine="python", on_bad_lines="warn")
        file_col = _find_column(transect.columns, "file")
        location_col = _find_column(transect.columns, "location")
        snow_depth_col = _find_column(transect.columns, "snow")

        for _, row in transect.iterrows():
            if pd.isna(row[file_col]):
                continue

            for pnt_path in _find_files(raw_dir, row[file_col], ".[pP][nN][tT]"):
                try:
                    profile = load_profile(pnt_path, station_dir.name, snowpit_root)
                except (ValueError, IndexError) as error:
                    # ValueError: load_profile's own >50%-negative-force guard.
                    # IndexError: snowmicropyn's ground/surface detection can
                    # raise this on a handful of genuinely unusual profiles
                    # (e.g. no discernible ground signal) rather than failing
                    # cleanly. Either way, one bad profile shouldn't abort
                    # the rest of the campaign.
                    print(f"skipping {pnt_path.name}: {type(error).__name__}: {error}")
                    continue
                training = bool(_find_files(raw_dir, row[file_col], ".ini"))
                snow_depth = pd.to_numeric(row[snow_depth_col], errors="coerce")
                location = pd.to_numeric(row[location_col], errors="coerce")
                profile = profile.assign({
                    "snow_depth": xr.DataArray(snow_depth),
                    "group_id": xr.DataArray(transect_dir.name),
                    "location": xr.DataArray(location),
                    "training": xr.DataArray(training),
                })
                profiles.append(profile)

    profiles = _pad_to_common_length(profiles)
    campaign = xr.concat(profiles, dim="profile")
    campaign = _drop_duplicate_profiles(campaign)
    return _add_attrs(campaign)


def _pad_to_common_length(profiles):
    """Pad every profile's `sample` dimension to the longest profile's length.

    `sample` carries no coordinate, so xr.concat cannot align mismatched
    lengths on its own; padding with NaN first keeps each profile's own
    values at their original positions.
    """
    max_length = max(profile.sizes["sample"] for profile in profiles)
    return [
        profile if profile.sizes["sample"] == max_length
        else profile.pad(sample=(0, max_length - profile.sizes["sample"]),
                         constant_values=np.nan)
        for profile in profiles
    ]


def _add_attrs(ds):
    """Attach CF-ish variable descriptions."""
    attrs = {
        "depth": dict(long_name="snow depth", units="mm",
                     description="distance from the air-snow interface"),
        "force": dict(long_name="snow resistance force", units="N"),
        "label": dict(long_name="snow type class", description=(
            "integers matching raw_import.LABEL_CODES: unknown=0, surface=1, "
            "ground=2, new snow=3, wind slab=4, fragmented & rounded=5, "
            "faceted=6, depth hoar=7, melt forms=8, ice forms=9, surface hoar=10")),
        "K2020b_density": dict(
            long_name="density from the King et al. (2020b) retrieval", units="kg/m^3"),
        "CR2020_density": dict(
            long_name="density from the Calonne & Richter (2020) retrieval", units="kg/m^3"),
        "CR2020_ssa": dict(
            long_name="specific surface area from the Calonne & Richter (2020) retrieval",
            units="m^2/kg"),
        "icethickness": dict(long_name="thickness of the ice floe", units="cm"),
        "freeboard": dict(long_name="sea ice freeboard", units="cm"),
        "name": dict(long_name="name of the original SMP file"),
        "snow_depth": dict(long_name="manually measured snow depth", units="cm"),
        "group_id": dict(long_name="transect this profile was measured in"),
        "location": dict(long_name="position of the profile along its transect", units="m"),
        "training": dict(long_name="whether this profile has hand-drawn layer markers"),
        "time": dict(long_name="time of SMP profile sampling"),
        "lat": dict(long_name="latitude", units="degrees_north"),
        "lon": dict(long_name="longitude", units="degrees_east"),
        "icetype": dict(long_name="type of sea ice floe"),
    }
    for var, var_attrs in attrs.items():
        if var in ds.variables:
            ds[var].attrs = var_attrs
    return ds

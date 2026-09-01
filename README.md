# SMP Grain-Type Classifier

Snow grain-type classification from **SnowMicroPen (SMP)** penetration-force
profiles, using a 1D convolutional neural network.

The SMP records penetration resistance at sub-millimetre resolution as it is
driven into a snowpack. This repository trains a CNN to read that force signal
and label **every depth bin** of a profile, turning a raw force trace into a
layer-by-layer stratigraphy.

| Class | Grain type |
|:-----:|------------|
| 0 | Wind Slab |
| 1 | Rounded & Faceted |
| 2 | Depth Hoar |
| 3 | Melt Forms |

The published model reaches **76.9 % accuracy** and **0.74 macro F1** on the
held-out validation profiles.

## Layout

```
smp-cnn/
├── data/
│   ├── labelled/smp_profiles_labelled.nc   # 153 hand-labelled profiles
│   ├── unlabelled/PS111_profiles.nc        # example campaign data
│   └── raw/                                # fine-resolution campaign files (not in version control)
├── models/
│   ├── cnn_grain_classifier.pth            # published weights
│   └── normalization_constants.pkl         # the constants they were trained with
├── notebooks/
│   ├── 00_preprocessing.ipynb              # raw SMP data -> CNN input files
│   ├── 01_training.ipynb                   # train and evaluate
│   ├── 02_inference.ipynb                  # classify unseen profiles
│   ├── 03_validation.ipynb                 # published model vs. hand labels
│   └── 04_remap_predictions.ipynb          # predictions -> fine-resolution depth grid
└── src/
    ├── config.py             # features, labels, paths, hyper-parameters
    ├── data.py               # splitting, normalization, the Dataset
    ├── model.py               # the CNN
    ├── training.py            # training loop
    ├── evaluation.py          # metrics
    ├── inference.py           # prediction on unseen profiles
    ├── plotting.py            # figures
    ├── raw_import.py          # raw .pnt + snowpit meta -> fine-resolution NetCDF
    ├── feature_engineering.py # fine-resolution -> depth-bin feature NetCDF
    └── remapping.py           # coarse predictions -> fine-resolution depth grid
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone <repository-url>
cd smp-cnn
python -m venv .venv
```

Activate it — on Linux or macOS:

```bash
source .venv/bin/activate
```

or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

There is nothing to install beyond the requirements: the notebooks add `src/` to
the path themselves with `sys.path.append("../src")`, so run them from the
`notebooks/` directory.

## Usage

Run the notebooks in order, or jump straight to the one you need:

- **`01_training.ipynb`** — trains a model from scratch on the labelled
  profiles and evaluates it on a held-out test split. Writes weights and
  normalization constants to `checkpoints/`.
- **`02_inference.ipynb`** — classifies profiles that have no labels. Point
  `input_path` at your own NetCDF file; results are written back as a
  `predicted_labels` variable.
- **`03_validation.ipynb`** — runs the published model over the labelled
  profiles and compares its predictions against the hand labels.

## Input data format

A NetCDF file readable by `xarray`, with dimensions `(profile, depth_bins)` and
these 18 feature variables:

```
mean_force, var_force, min_force, max_force, distance,
first_derivative, second_derivative,
first_absolute_derivative, second_absolute_derivative,
mean_force_rolled, var_force_rolled, min_force_rolled, max_force_rolled,
force_median, lambda, f0, delta, L
```

`lambda`, `f0`, `delta` and `L` are the micro-mechanical parameters of the
Löwe & van Herwijnen (2012) shot-noise model, computed with
[snowmicropyn](https://snowmicropyn.readthedocs.io).

Labelled files additionally carry `dominant_label` with the dominant grain type
per depth bin, stored in a five-class scheme; `data.merge_to_four_classes`
combines *Fragmented & Rounded* with *Faceted*, which the force signal alone
cannot reliably separate, giving the four classes above.

Depth bins where any feature is `NaN` are dropped automatically. Profiles left
with no valid bins are skipped and reported.

## Preprocessing

Turns raw SMP field data into the file format above, and maps predictions back
onto the original depth grid afterwards. Needs
[snowmicropyn](https://snowmicropyn.readthedocs.io) and `scipy`, listed
separately in `requirements-preprocessing.txt` since training, evaluation and
inference don't need them:

```bash
python -m pip install -r requirements.txt -r requirements-preprocessing.txt
```

- **`00_preprocessing.ipynb`** — walks one campaign's raw `.pnt` profiles and
  snowpit meta (`src/raw_import.py`), builds the fine-resolution
  `(profile, sample)` campaign file, then aggregates it into 1 mm depth-bin
  features (`src/feature_engineering.py`) matching the format above. Writes
  the fine-resolution file to `data/raw/` (excluded from version control — as
  large as the source data) and the feature file to `data/unlabelled/` or
  `data/labelled/`. The fine-resolution file also carries King et al. (2020b)
  and Calonne & Richter (2020) density/SSA retrievals, so it's useful on its
  own — before, or entirely without, running it through the CNN.
- **`04_remap_predictions.ipynb`** — takes the fine-resolution file from
  `00_preprocessing.ipynb` and the `predicted_labels` written by
  `02_inference.ipynb`, and assigns every raw depth sample the label of the
  1 mm bin it falls into (`src/remapping.py`), so the CNN's coarse
  predictions can be plotted or analyzed at the SMP's native resolution.

A campaign directory is expected in the layout the SMP field software writes:
one folder per station (e.g. `PS111_20180211_3-2_SMP`), each with one or more
`*TRANS*` transect folders holding a tab-separated `*_ini.txt` table and a
`_raw`/`raw` folder of `.pnt` profiles (some campaigns keep one shared
`_raw`/`raw` per station instead — both are checked), plus a matching
snowpit directory with one `*META.txt` per station. The transect table's
columns are detected by keyword rather than a fixed layout, since campaigns
name and order them differently (a bare file number vs. a full profile name,
for instance). Validated against four campaigns' raw data (AFIN25, PS111,
PS118, PS124), each laid out slightly differently.

## Model

Three 1D convolution blocks (18 → 32 → 64 → 128 channels, kernel size 21, each
followed by ReLU and GroupNorm), dropout, then a 1×1 convolution that
classifies each depth bin from its local context. 228,324 parameters.

Padding is `kernel_size // 2`, so the sequence length is preserved and a profile
of any length yields exactly one prediction per depth bin.

The published checkpoint was trained with **kernel size 21**, which is what
`config.KERNEL_SIZE` holds. Loading it with a different kernel size fails on a
shape mismatch.

## Reproducibility

- Profiles are split as whole units, never by depth bin, so bins from one
  profile cannot land in two splits. The shuffle is seeded
  (`config.SPLIT_SEED = 42`).
- Normalization constants and class weights are computed from the training
  profiles only.
- The constants travel with the weights, in
  `models/normalization_constants.pkl`. Inference must reuse them —
  recomputing from a new campaign rescales the inputs away from what the model
  learned and quietly degrades predictions instead of failing outright.
- Class frequencies are imbalanced, so the loss is weighted by inverse
  frequency.

### Data availability

`data/` holds the labelled training set and one example campaign (PS111) —
enough to train, validate and run inference end to end. The full-resolution
source profiles and the complete campaign predictions are too large for version
control and are available from the authors on request. The raw SMP datasets are available on PANGEA.

## License

The original code in this repository is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.

Third-party software used by this project remains subject to its respective
licenses. In particular, [snowmicropyn](https://github.com/slf-dot-ch/snowmicropyn)
is distributed under its own GPL license and is used here only as an external
dependency during preprocessing; no snowmicropyn source code is included or
modified in this repository.

## Affiliation

Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research (AWI).

## Acknowledgements

SMP measurements were collected during the PS111, PS118 and PS124 Polarstern
expeditions. Micro-mechanical profile parameters are derived using
[snowmicropyn](https://snowmicropyn.readthedocs.io).

Several functions in `src/raw_import.py` and `src/feature_engineering.py` (raw
force-signal cleanup, layer-marker labelling, and the depth-bin/shot-noise
feature aggregation) are adapted from Julia Kaltenborn's
[snowdragon](https://github.com/liellnima/snowdragon) (MIT License), credited
inline at each function.

## Citation

The citation for the accompanying publication will be added once the paper is
available.

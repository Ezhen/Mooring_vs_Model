# nsval — North Sea Validation Toolkit

Point-based extraction, climatological analysis, and model–observation
validation for CMEMS in-situ archives and ROMS ocean model output.

---

## Installation

### On a supercomputer (conda environment, no internet on compute nodes)

```bash
# 1. transfer the package
scp nsval.zip user@cluster:~/

# 2. unpack and build structure
cd ~/Validation/field_vs_model
unzip ~/nsval.zip

# 3. activate your environment
conda activate Yoda   # or your environment name

# 4. create pyproject.toml
cat > nsval_pkg/pyproject.toml << 'TOML'
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
TOML

# 5. install
pip install nsval_pkg/ --no-deps --no-build-isolation

# 6. verify
python -c "import nsval; print(nsval.__version__)"
```

### Standard install (internet available)

```bash
pip install -e .           # development install
pip install -e ".[full]"   # include cftime + netCDF4
```

### Adding or updating a module after install

```bash
SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")
cp new_module.py $SITEPKG/nsval/
# clear cache
find $SITEPKG/nsval -name "*.pyc" -delete
```

---

## Package structure

```
nsval/
├── __init__.py           Public API and aliases
├── inventory.py          NetCDF archive inventory and variable catalogue
├── nsval_inspect.py      Dataset inspection (memory, search, QC)
├── utils.py              Shared helpers (haversine, time decoding, I/O)
├── intake/
│   ├── cmems.py          Extract point timeseries from CMEMS in-situ files
│   └── roms.py           Extract point timeseries from ROMS AVG files
├── analyse/
│   └── timeseries.py     Timeseries diagnostics, climatology, figures
└── validate/
    ├── metrics.py        Core metric functions (shared)
    ├── daily.py          Nearest-timestep model vs obs validation
    ├── monthly.py        Monthly-mean model vs obs validation
    ├── vertical.py       CMEMS vertical profile extraction and heatmaps
    └── vertical_model.py Collocated obs vs ROMS vertical validation
```

---

## Workflow

```
Archive (.nc files)                        ROMS AVG files
        │                                       │
nsval.scan_dataset()                            │
nsval.find_variable()                           │
nsval.estimate_memory_load()                    │
nsval.qc_summary()                             │
        │                                       │
nsval.safe_point_timeseries()       nsval.intake.roms.extract_point()
        │                                       │
        └──────────────── CSV ──────────────────┘
                           │
            nsval.analyse.timeseries     ← diagnostics on a single dataset
                           │
            nsval.compare_timeseries()   ← model vs obs, monthly means
            nsval.validate.daily         ← model vs obs, nearest timestep
                           │
            nsval.validate.vertical      ← time-depth heatmaps (obs)
            nsval.validate.vertical_model← collocated obs vs ROMS profiles
```

---

## Public API

All core functions are accessible directly from `nsval` after import:

```python
import nsval

# ── inspect ───────────────────────────────────────────────────────
nsval.scan_dataset()              # inventory of all .nc files in a folder
nsval.build_variable_catalogue()  # unique variables + metadata
nsval.find_variable()             # keyword search across catalogue
nsval.estimate_memory_load()      # estimate array size before loading
nsval.qc_summary()                # CMEMS QC flag breakdown

# ── extract ───────────────────────────────────────────────────────
nsval.safe_point_timeseries()     # CMEMS point timeseries extraction
nsval.intake.roms.extract_point() # ROMS point timeseries extraction

# ── analyse ───────────────────────────────────────────────────────
nsval.analyse.timeseries.analyse()# timeseries + climatology + metrics

# ── validate ──────────────────────────────────────────────────────
nsval.compare_timeseries()        # monthly model vs obs validation
nsval.validate.daily.validate()   # nearest-timestep model vs obs
nsval.validate.vertical.analyse_vertical()        # obs time-depth
nsval.validate.vertical_model.validate_vertical() # collocated profiles
```

---

## Step-by-step usage

### Step 1 — Inspect the archive

```python
import nsval

# check what's in the folder
nsval.scan_dataset(
    folder  = "/path/to/MOORING_DATA",
    out_csv = "netcdf_inventory.csv",
)

# build variable catalogue
nsval.build_variable_catalogue(
    inventory_csv = "netcdf_inventory.csv",
    out_csv       = "unique_variables.csv",
)

# search for a variable
nsval.find_variable("temp", catalogue_csv="unique_variables.csv")
nsval.find_variable("nitrate", catalogue_csv="unique_variables.csv")

# check memory before loading
nsval.estimate_memory_load(
    folder   = "/path/to/MOORING_DATA",
    variable = "TEMP",
    warn_gb  = 1.0,
)

# QC overview
nsval.qc_summary(
    folder   = "/path/to/MOORING_DATA",
    variable = "TEMP",
)
```

### Step 2 — Extract observations

```python
nsval.safe_point_timeseries(
    folder        = "/path/to/MOORING_DATA",
    variable      = "TEMP",
    lat0          = 54.5,
    lon0          = 4.0,
    radius_km     = 50,
    vertical_mode = "surface",
    out_csv       = "TEMP_scoop_54.5_4.0.csv",
)
```

### Step 3 — Extract model

```python
from nsval.intake.roms import extract_point

extract_point(
    simulations = {
        "sim_2006": {
            "folder" : "/scratch/.../CE2COAST_2006",
            "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
        },
        "sim_2009": {
            "folder" : "/scratch/.../CE2COAST_2009",
            "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
        },
    },
    variable = "temp",
    lat0     = 54.5,
    lon0     = 4.0,
    s_level  = -1,
    out_csv  = "roms_temp_54.5_4.0.csv",
)
```

### Step 4 — Analyse each dataset independently

```python
from nsval.analyse.timeseries import analyse

analyse(csv_file="TEMP_scoop_54.5_4.0.csv", variable="TEMP",
        qc_col="TEMP_QC", flag="Archive")

analyse(csv_file="roms_temp_54.5_4.0.csv", variable="temp_celsius",
        qc_col=None, flag="Model")
```

### Step 5 — Validate model vs observations

```python
# monthly matching (recommended for sparse obs)
nsval.compare_timeseries(
    obs_csv   = "TEMP_scoop_54.5_4.0.csv",
    model_csv = "roms_temp_54.5_4.0.csv",
    obs_var   = "TEMP",
    model_var = "temp_celsius",
)

# daily nearest-timestep matching
from nsval.validate.daily import validate as validate_daily
validate_daily(
    obs_csv   = "TEMP_scoop_54.5_4.0.csv",
    model_csv = "roms_temp_54.5_4.0.csv",
    obs_var   = "TEMP",
    model_var = "temp_celsius",
)
```

### Step 6 — Vertical validation

```python
# extract observed vertical profiles
from nsval.validate.vertical import analyse_vertical

analyse_vertical(
    folder     = "/path/to/MOORING_DATA",
    variable   = "TEMP",
    lat0       = 54.5,
    lon0       = 4.0,
    radius_km  = 100,
    min_years  = 1,
    depth_max  = 200,
    out_csv    = "vertical_TEMP_54.5_4.0.csv",
    out_figure = "vertical_TEMP_54.5_4.0.png",
)

# collocate with ROMS and validate depth-by-depth
from nsval.validate.vertical_model import validate_vertical

validate_vertical(
    obs_csv       = "vertical_TEMP_54.5_4.0.csv",
    roms_folder   = "/scratch/.../CE2COAST_2006",
    roms_pattern  = "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
    roms_variable = "temp",
    obs_variable  = "TEMP",
    space_km      = 20.0,
    time_days     = 1.0,
    depth_max     = 150.0,
    depth_bins    = [0, 5, 10, 20, 30, 50, 75, 100, 150],
    out_csv       = "vertical_validation.csv",
    out_figure    = "vertical_validation.png",
)
```

---

## Validation metrics

All validators compute the same 26 metrics:

| Group | Metrics |
|---|---|
| Sample | n, mean, std, std ratio |
| Error | Bias, MAE, MSE, RMSE, centred RMSE, normalised RMSE, max error, MAPE, P10/P50/P90 errors, hit rates ±0.5°C and ±1°C |
| Skill | Pearson r + p, R², Spearman ρ + p, IoA, modified IoA, NSE, KGE, skill score |

Plus full seasonal breakdown (DJF / MAM / JJA / SON), monthly climatological
bias table, and — for vertical validation — all metrics per depth bin.

---

## Figures produced

| Function | Figures |
|---|---|
| `analyse()` | Timeseries + anomalies, monthly climatology, day-of-year climatology |
| `validate()` daily | Scatter (KDE), Taylor diagram, monthly bias, Q-Q, residuals timeseries |
| `validate()` monthly | Scatter (by month), Taylor diagram, climatology + bias, Q-Q, timeseries, seasonal boxplots |
| `analyse_vertical()` | Time-depth heatmap + mean profile per station |
| `validate_vertical()` | Obs vs model heatmaps, bias heatmap, mean profiles, depth-resolved bias, metrics table |

---

## Dependencies

`numpy` · `pandas` · `xarray` · `scipy` · `matplotlib`
Optional: `cftime`, `netCDF4` (for non-standard time calendars)

---

## Changelog

See `CHANGELOG.md` for version history.
To add an entry from the command line:
```bash
log "description of what changed"
```
(requires the `log` alias defined in `~/.bashrc`)

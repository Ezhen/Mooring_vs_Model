# nsval — North Sea Validation Toolkit

Point-based and gridded extraction, climatological analysis, and
model–observation validation for CMEMS in-situ archives, ROMS ocean
model output, and satellite SST products.

---

## Installation

### On a supercomputer (conda environment, no internet on compute nodes)

```bash
# 1. transfer the package
scp nsval.zip user@cluster:~/

# 2. unpack
cd ~/Validation/field_vs_model
unzip ~/nsval.zip

# 3. activate your environment
conda activate Yoda

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
pip install -e .            # development install
pip install -e ".[full]"    # include cftime + netCDF4
```

### Adding or updating a module after install

```bash
SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")
cp new_module.py $SITEPKG/nsval/validate/   # or intake/ or analyse/
find $SITEPKG/nsval -name "*.pyc" -delete
```

### Changelog (one-line command)

```bash
log "description of what changed"   # requires alias in ~/.bashrc
```

---

## Package structure

```
nsval/
├── __init__.py              Public API and aliases
├── inventory.py             NetCDF archive inventory and variable catalogue
├── nsval_inspect.py         Dataset inspection (memory, search, QC)
├── utils.py                 Shared helpers (haversine, time decoding, I/O)
├── intake/
│   ├── cmems.py             Extract point timeseries from CMEMS in-situ files
│   ├── cmems_region.py      Extract all stations inside a bounding box
│   └── roms.py              Extract point timeseries from ROMS AVG files
├── analyse/
│   ├── timeseries.py        Timeseries diagnostics, climatology, figures
│   └── seasonal_dashboard.py  4-season dashboard (obs + model overlaid)
└── validate/
    ├── metrics.py           Core metric functions (26 metrics, shared)
    ├── daily.py             Nearest-timestep model vs obs validation
    ├── monthly.py           Monthly-mean model vs obs validation
    ├── vertical.py          CMEMS vertical profile extraction and heatmaps
    ├── vertical_model.py    Collocated obs vs ROMS vertical validation
    ├── spatial.py           Spatial bias/RMSE/target maps (cartopy)
    └── sst_satellite.py     Gridded ROMS vs satellite SST validation
```

---

## Workflow

```
Archive (.nc files)                        ROMS AVG files        Satellite SST
        │                                       │                      │
nsval.scan_dataset()                            │                      │
nsval.find_variable()                           │                      │
nsval.estimate_memory_load()                    │                      │
nsval.qc_summary()                             │                      │
        │                                       │                      │
nsval.safe_point_timeseries()       nsval.intake.roms.extract_point()  │
nsval.intake.cmems_region           (point or domain-wide)             │
  .scoop_region()                                                       │
        │                                       │                      │
        └──────────── CSV ───────────────────────┘                     │
                       │                                               │
    nsval.analyse.timeseries          ← single-dataset diagnostics     │
    nsval.analyse.seasonal_dashboard  ← obs + model seasonal overlay   │
                       │                                               │
    nsval.compare_timeseries()        ← monthly model vs obs           │
    nsval.validate.daily              ← nearest-timestep model vs obs  │
    nsval.validate.vertical           ← obs time-depth heatmaps        │
    nsval.validate.vertical_model     ← collocated depth profiles      │
    nsval.validate.spatial            ← domain-wide spatial maps       │
                                                                       │
    nsval.validate.sst_satellite  ←────────────────────────────────────┘
                                       gridded ROMS vs satellite SST
```

---

## Public API

All core functions accessible directly from `nsval` after import:

```python
import nsval

# ── inspect ───────────────────────────────────────────────────────
nsval.scan_dataset()              # inventory of all .nc files in a folder
nsval.build_variable_catalogue()  # unique variables + metadata
nsval.find_variable()             # keyword search across catalogue
nsval.estimate_memory_load()      # estimate array size before loading
nsval.qc_summary()                # CMEMS QC flag breakdown

# ── extract ───────────────────────────────────────────────────────
nsval.safe_point_timeseries()               # CMEMS point timeseries (radius)
nsval.intake.cmems_region.scoop_region()    # CMEMS domain-wide extraction
nsval.intake.roms.extract_point()           # ROMS point timeseries

# ── analyse ───────────────────────────────────────────────────────
nsval.analyse.timeseries.analyse()                      # timeseries + climatology
nsval.analyse.seasonal_dashboard.seasonal_dashboard()   # seasonal dashboard

# ── validate — point / timeseries ────────────────────────────────
nsval.compare_timeseries()           # monthly model vs obs
nsval.validate.daily.validate()      # nearest-timestep model vs obs

# ── validate — vertical ───────────────────────────────────────────
nsval.validate.vertical.analyse_vertical()         # obs time-depth
nsval.validate.vertical_model.validate_vertical()  # collocated profiles

# ── validate — spatial / gridded ──────────────────────────────────
nsval.validate.spatial.validate_spatial()          # bias/RMSE/target maps
nsval.validate.sst_satellite.validate_sst()        # gridded SST validation
```

---

## Step-by-step usage

### Step 1 — Inspect the archive

```python
import nsval

nsval.scan_dataset(folder="/path/to/MOORING_DATA", out_csv="inventory.csv")
nsval.build_variable_catalogue(inventory_csv="inventory.csv",
                                out_csv="unique_variables.csv")
nsval.find_variable("temp",    catalogue_csv="unique_variables.csv")
nsval.find_variable("nitrate", catalogue_csv="unique_variables.csv")
nsval.estimate_memory_load(folder="/path/to/MOORING_DATA", variable="TEMP")
nsval.qc_summary(folder="/path/to/MOORING_DATA", variable="TEMP")
```

### Step 2 — Extract observations

```python
# point extraction (radius around a target)
nsval.safe_point_timeseries(
    folder="MOORING_DATA", variable="TEMP",
    lat0=54.5, lon0=4.0, radius_km=50,
    vertical_mode="surface", out_csv="TEMP_scoop_54.5_4.0.csv",
)

# domain-wide extraction (full North Sea)
from nsval.intake.cmems_region import scoop_region
scoop_region(
    folder="MOORING_DATA", variable="TEMP",
    lat_min=51.0, lat_max=62.0, lon_min=-4.0, lon_max=10.0,
    out_csv="TEMP_NorthSea.csv",
)
```

### Step 3 — Extract model

```python
from nsval.intake.roms import extract_point

extract_point(
    simulations={
        "sim_2006": {"folder": "/scratch/.../CE2COAST_2006",
                     "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc"},
        "sim_2009": {"folder": "/scratch/.../CE2COAST_2009",
                     "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc"},
    },
    variable="temp", lat0=54.5, lon0=4.0, s_level=-1,
    out_csv="roms_temp_54.5_4.0.csv",
)
```

### Step 4 — Analyse each dataset

```python
from nsval.analyse.timeseries import analyse

analyse(csv_file="TEMP_scoop_54.5_4.0.csv", variable="TEMP",
        qc_col="TEMP_QC", flag="Archive")
analyse(csv_file="roms_temp_54.5_4.0.csv", variable="temp",
        qc_col=None, flag="Model")
```

### Step 4b — Seasonal dashboard

```python
from nsval.analyse.seasonal_dashboard import seasonal_dashboard

seasonal_dashboard(
    obs_csv="TEMP_scoop_54.5_4.0.csv", obs_var="TEMP",
    model_csv="roms_temp_54.5_4.0.csv", model_var="temp",
    out_prefix="examples/outputs/seasonal_dashboard",
)
```

### Step 5 — Validate model vs observations (timeseries)

```python
# monthly matching
nsval.compare_timeseries(
    obs_csv="TEMP_scoop_54.5_4.0.csv", model_csv="roms_temp_54.5_4.0.csv",
    obs_var="TEMP", model_var="temp",
)

# nearest-timestep matching
from nsval.validate.daily import validate as validate_daily
validate_daily(
    obs_csv="TEMP_scoop_54.5_4.0.csv", model_csv="roms_temp_54.5_4.0.csv",
    obs_var="TEMP", model_var="temp",
)
```

### Step 6 — Vertical validation

```python
from nsval.validate.vertical import analyse_vertical
from nsval.validate.vertical_model import validate_vertical

analyse_vertical(
    folder="MOORING_DATA", variable="TEMP",
    lat0=54.5, lon0=4.0, radius_km=100, min_years=1, depth_max=200,
    out_csv="vertical_TEMP.csv", out_figure="vertical_TEMP.png",
)

validate_vertical(
    obs_csv="vertical_TEMP.csv",
    roms_folder="/scratch/.../CE2COAST_2006",
    roms_pattern="Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
    roms_variable="temp", obs_variable="TEMP",
    space_km=20.0, time_days=1.0, depth_max=150.0,
    depth_bins=[0, 5, 10, 20, 30, 50, 75, 100, 150],
    out_csv="vertical_validation.csv", out_figure="vertical_validation.png",
)
```

### Step 7 — Spatial validation (domain-wide, cartopy maps)

```python
from nsval.validate.spatial import validate_spatial

for season in ["DJF", "MAM", "JJA", "SON"]:
    validate_spatial(
        obs_csv="TEMP_NorthSea.csv",
        roms_folder="/scratch/.../CE2COAST_2006",
        roms_pattern="Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
        season=season,
        out_csv=f"examples/outputs/spatial_{season}.csv",
        out_prefix=f"examples/outputs/spatial_{season}",
        show_figures=False,
    )
```

### Step 8 — Satellite SST validation (gridded)

```python
from nsval.validate.sst_satellite import validate_sst

validate_sst(
    roms_folder  = "/scratch/.../CE2COAST_2006",
    roms_pattern = "Hindcast_CE2COAST_AVG_{year}_2c_era5_bcorr.nc",
    sat_folder   = "/path/to/Satellite_SST_replotted",
    sat_pattern  = "DMI_BAL_SST_L4_REP_OBSERVATIONS_{year}_sst.nc",
    years        = list(range(1993, 2021)),
    out_prefix   = "examples/outputs/sst_validation",
)
```

---

## Validation metrics

### Point / timeseries validators — 26 metrics

| Group | Metrics |
|---|---|
| Sample | n, mean, std, std ratio |
| Error | Bias, MAE, MSE, RMSE, centred RMSE, normalised RMSE, max error, MAPE, P10/P50/P90 errors, hit rates ±0.5°C and ±1°C |
| Skill | Pearson r + p, R², Spearman ρ + p, IoA, modified IoA, NSE, KGE, skill score |

Plus full seasonal breakdown (DJF/MAM/JJA/SON) and monthly bias table.

### Vertical validator — same 26 metrics per depth bin

### Spatial validator — per-station bias, RMSE, normalised metrics

### SST satellite validator — 8 gridded metric fields

| Field | Description |
|---|---|
| `bias` | Mean model − satellite (°C) |
| `rmse` | Root mean squared error (°C) |
| `mae` | Mean absolute error (°C) |
| `correlation` | Pearson r across time |
| `std_ratio` | Model std / satellite std (>1 = too variable) |
| `hit_rate` | Fraction of timesteps with \|bias\| < 1°C |
| `bias_trend` | Linear trend of annual mean bias (°C/year) |
| `amp_error` | Seasonal amplitude error JJA−DJF (°C) |

---

## Figures produced

| Function | Figures |
|---|---|
| `analyse()` | Timeseries + anomalies, monthly climatology, day-of-year climatology |
| `seasonal_dashboard()` | Seasonal maps, seasonal timeseries, climatological rose |
| `validate()` daily | Scatter (KDE), Taylor diagram, monthly bias, Q-Q, residuals timeseries |
| `validate()` monthly | Scatter (by month), Taylor diagram, climatology + bias, Q-Q, timeseries, seasonal boxplots |
| `analyse_vertical()` | Time-depth heatmap + mean profile per station |
| `validate_vertical()` | Obs vs model heatmaps, bias heatmap, mean profiles, depth-resolved bias, metrics table |
| `validate_spatial()` | Bias map (cartopy), RMSE map, target diagram, obs vs model scatter |
| `validate_sst()` | Seasonal bias maps (cartopy), 6-panel metric dashboard, per-year timeseries |

---

## Dependencies

**Required:** `numpy` · `pandas` · `xarray` · `scipy` · `matplotlib` · `cartopy`

**Optional:** `cftime` · `netCDF4` (for non-standard ROMS time calendars)

---

## Changelog

See `CHANGELOG.md` for version history.
To add an entry from the command line:
```bash
log "nsval 0.2.0: added spatial, sst_satellite, seasonal_dashboard, scoop_region"
```

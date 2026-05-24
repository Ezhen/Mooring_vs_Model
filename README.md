# nsval — North Sea Validation Toolkit

Point-based extraction, climatological analysis, and model–observation
validation for CMEMS in-situ archives and ROMS ocean model output.

---

## Installation

```bash
pip install -e .                  # development install
pip install -e ".[full]"          # include cftime + netCDF4
```

---

## Package structure

```
nsval/
├── inventory.py          NetCDF archive inventory and variable catalogue
├── utils.py              Shared helpers (haversine, time decoding, I/O)
├── intake/
│   ├── cmems.py          Extract point timeseries from CMEMS in-situ files
│   └── roms.py           Extract point timeseries from ROMS AVG files
├── analyse/
│   └── timeseries.py     Timeseries diagnostics, climatology, figures
└── validate/
    ├── metrics.py        Core metric functions (shared)
    ├── daily.py          Nearest-timestep model vs obs validation
    └── monthly.py        Monthly-mean model vs obs validation
```

---

## Workflow

```
Archive (.nc files)                 ROMS AVG files
        │                                  │
nsval.inventory                            │
        │                                  │
nsval.intake.cmems              nsval.intake.roms
        │                                  │
        └──────────── CSV ─────────────────┘
                       │
          nsval.analyse.timeseries     ← diagnostics on a single dataset
                       │
          nsval.validate.daily         ← model vs obs, nearest timestep
          nsval.validate.monthly       ← model vs obs, monthly means
```

---

## Python API

### 1. Build archive inventory

```python
from nsval.inventory import build_inventory, build_variable_catalogue

build_inventory(folder="/path/to/MOORING_DATA", out_csv="inventory.csv")
build_variable_catalogue(inventory_csv="inventory.csv",
                         out_csv="variables.csv")
```

### 2. Extract CMEMS in-situ timeseries

```python
from nsval.intake.cmems import scoop_point

scoop_point(
    folder        = "/path/to/MOORING_DATA",
    variable      = "TEMP",
    lat0          = 54.5,
    lon0          = 4.0,
    radius_km     = 50,
    vertical_mode = "surface",
    out_csv       = "TEMP_scoop_54.5_4.0.csv",
)
```

### 3. Extract ROMS model timeseries

```python
from nsval.intake.roms import extract_point

extract_point(
    simulations = {
        "sim_2006": {"folder": "/scratch/.../CE2COAST_2006",
                     "pattern": "Hindcast_CE2COAST_AVG_*.nc"},
        "sim_2009": {"folder": "/scratch/.../CE2COAST_2009",
                     "pattern": "Hindcast_CE2COAST_AVG_*.nc"},
    },
    variable = "temp",
    lat0     = 54.5,
    lon0     = 4.0,
    s_level  = -1,
    out_csv  = "roms_temp_54.5_4.0.csv",
)
```

### 4. Analyse a single timeseries

```python
from nsval.analyse.timeseries import analyse

analyse(
    csv_file = "TEMP_scoop_54.5_4.0.csv",
    variable = "TEMP",
    qc_col   = "TEMP_QC",
    flag     = "Archive",
)
```

### 5. Validate model vs observations

```python
from nsval.validate.monthly import validate

validate(
    obs_csv   = "TEMP_scoop_54.5_4.0.csv",
    model_csv = "roms_temp_54.5_4.0.csv",
    obs_var   = "TEMP",
    model_var = "temp_celsius",
)
```

---

## Command-line interface

Every module exposes a CLI entry point after `pip install -e .`:

```bash
nsval-inventory  --folder /path/to/data
nsval-cmems      --folder /path/to/data --variable TEMP --lat 54.5 --lon 4.0
nsval-roms       --folders /path/sim1 /path/sim2 \
                 --labels sim_2006 sim_2009 \
                 --variable temp --lat 54.5 --lon 4.0
nsval-analyse    --csv TEMP_scoop_54.5_4.0.csv --variable TEMP --flag Archive
nsval-validate-daily    --obs TEMP_scoop.csv --model roms_temp.csv
nsval-validate-monthly  --obs TEMP_scoop.csv --model roms_temp.csv
```

---

## Validation metrics

Both validators compute the same 26 metrics:

| Group | Metrics |
|---|---|
| Sample | n, mean, std, std ratio |
| Error | Bias, MAE, MSE, RMSE, centred RMSE, normalised RMSE, max error, MAPE, P10/P50/P90 errors, hit rates ±0.5°C and ±1°C |
| Skill | Pearson r + p, R², Spearman ρ + p, IoA, modified IoA, NSE, KGE, skill score |

Plus full seasonal breakdown (DJF / MAM / JJA / SON) and monthly climatological bias table.

---

## Dependencies

`numpy` · `pandas` · `xarray` · `scipy` · `matplotlib`  
Optional: `cftime`, `netCDF4` (for non-standard time calendars)

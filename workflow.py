# ── STEP 1: inventory ─────────────────────────────────────────────
from nsval.inventory import build_inventory, build_variable_catalogue

build_inventory(
    folder  = "/home/ulg/mast/eivanov/Validation/MOORING_DATA/",
    out_csv = "netcdf_inventory.csv",
)

build_variable_catalogue(
    inventory_csv = "netcdf_inventory.csv",
    out_csv       = "unique_variables.csv",
)

# ── STEP 2: extract observations ──────────────────────────────────
from nsval.intake.cmems import scoop_point

scoop_point(
    folder        = "/home/ulg/mast/eivanov/Validation/MOORING_DATA/",
    variable      = "TEMP",
    lat0          = 54.5,
    lon0          = 4.0,
    radius_km     = 50,
    vertical_mode = "surface",
    out_csv       = "TEMP_scoop_54.5_4.0.csv",
)

# ── STEP 3: extract model ──────────────────────────────────────────
from nsval.intake.roms import extract_point

extract_point(
    simulations = {
        "sim_2006": {
            "folder" : "/scratch/ulg/mast/eivanov/Output/CE2COAST_2006",
            "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
        },
        "sim_2009": {
            "folder" : "/scratch/ulg/mast/eivanov/Output/CE2COAST_2009",
            "pattern": "Hindcast_CE2COAST_AVG_*_2c_atm3.nc",
        },
    },
    variable = "temp",
    lat0     = 54.5,
    lon0     = 4.0,
    s_level  = -1,
    out_csv  = "roms_temp_54.5_4.0.csv",
)

# ── STEP 4: analyse each dataset independently ─────────────────────
from nsval.analyse.timeseries import analyse

analyse(
    csv_file = "TEMP_scoop_54.5_4.0.csv",
    variable = "TEMP",
    qc_col   = "TEMP_QC",
    flag     = "Archive",
)

analyse(
    csv_file = "roms_temp_54.5_4.0.csv",
    variable = "temp_celsius",
    qc_col   = None,
    flag     = "Model",
)

# ── STEP 5: validate model vs observations ─────────────────────────
from nsval.validate.daily import validate as validate_daily
from nsval.validate.monthly import validate as validate_monthly

# nearest-timestep matching
validate_daily(
    obs_csv   = "TEMP_scoop_54.5_4.0.csv",
    model_csv = "roms_temp_54.5_4.0.csv",
    obs_var   = "TEMP",
    model_var = "temp_celsius",
)

# monthly-mean matching
validate_monthly(
    obs_csv   = "TEMP_scoop_54.5_4.0.csv",
    model_csv = "roms_temp_54.5_4.0.csv",
    obs_var   = "TEMP",
    model_var = "temp_celsius",
)

from nsval.validate.vertical_model import validate_vertical

validate_vertical(
    obs_csv       = "vertical_TEMP_54.5_4.0.csv",  # from vertical.py
    roms_folder   = "/scratch/ulg/mast/eivanov/Output/CE2COAST_2006",
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

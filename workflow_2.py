from nsval.validate.vertical import analyse_vertical

analyse_vertical(
    folder      = "/home/ulg/mast/eivanov/Validation/MOORING_DATA/",
    variable    = "TEMP",
    lat0        = 54.5,
    lon0        = 4.0,
    radius_km   = 100,
    min_years   = 1,
    depth_max   = 200,
    out_csv     = "vertical_TEMP_54.5_4.0.csv",
    out_figure  = "vertical_TEMP_54.5_4.0.png",
)

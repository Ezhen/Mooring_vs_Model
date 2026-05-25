from setuptools import setup, find_packages

setup(
    name             = "nsval",
    version          = "0.1.0",
    description      = "North Sea Validation Toolkit — point extraction, "
                       "climatological analysis, and model–observation validation",
    author           = "E. Ivanov",
    #python_requires  = ">=3.9",
    packages=find_packages(where="nsval_pkg"),
    package_dir={"": "nsval_pkg"},
    install_requires = [
        "numpy>=1.23",
        "pandas>=1.5",
        "xarray>=2022.6",
        "scipy>=1.9",
        "matplotlib>=3.6",
    ],
    extras_require   = {
        "full": ["cftime", "netCDF4"],
    },
    entry_points     = {
        "console_scripts": [
            "nsval-inventory = nsval.inventory:_cli",
            "nsval-cmems     = nsval.intake.cmems:_cli",
            "nsval-roms      = nsval.intake.roms:_cli",
            "nsval-analyse   = nsval.analyse.timeseries:_cli",
            "nsval-validate-daily   = nsval.validate.daily:_cli",
            "nsval-validate-monthly = nsval.validate.monthly:_cli",
        ],
    },
)

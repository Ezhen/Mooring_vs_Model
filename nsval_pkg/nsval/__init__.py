"""nsval — North Sea Validation Toolkit"""
__version__ = "0.1.0"
__author__  = "E. Ivanov"
from . import inventory
from .intake import cmems, roms
from .analyse import timeseries
from .validate import metrics, daily, monthly, vertical, vertical_model

from .inspect import estimate_memory_load, find_variable, qc_summary
from .inventory import build_inventory as scan_dataset
from .inventory import build_variable_catalogue
from .intake.cmems import scoop_point as safe_point_timeseries
from .validate.monthly import validate as compare_timeseries
from .analyse import seasonal_dashboard

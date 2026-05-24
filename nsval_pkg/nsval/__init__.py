"""nsval — North Sea Validation Toolkit"""
__version__ = "0.1.0"
__author__  = "E. Ivanov"
from . import inventory
from .intake import cmems, roms
from .analyse import timeseries
from .validate import metrics, daily, monthly

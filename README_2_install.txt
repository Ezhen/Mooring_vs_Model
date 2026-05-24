# create structure
mkdir -p nsval_pkg/nsval/intake
mkdir -p nsval_pkg/nsval/analyse
mkdir -p nsval_pkg/nsval/validate

# move files
mv cmems.py       nsval_pkg/nsval/intake/
mv roms.py        nsval_pkg/nsval/intake/
mv timeseries.py  nsval_pkg/nsval/analyse/
mv metrics.py     nsval_pkg/nsval/validate/
mv daily.py       nsval_pkg/nsval/validate/
mv monthly.py     nsval_pkg/nsval/validate/
mv inventory.py   nsval_pkg/nsval/
mv utils.py       nsval_pkg/nsval/

# create __init__.py files
cat > nsval_pkg/nsval/__init__.py << 'EOF'
"""nsval — North Sea Validation Toolkit"""
__version__ = "0.1.0"
__author__  = "E. Ivanov"
from . import inventory
from .intake import cmems, roms
from .analyse import timeseries
from .validate import metrics, daily, monthly
EOF

echo "" > nsval_pkg/nsval/intake/__init__.py
echo "" > nsval_pkg/nsval/analyse/__init__.py
echo "" > nsval_pkg/nsval/validate/__init__.py

# copy directly to site-packages
SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Installing to: $SITEPKG"
cp -r nsval_pkg/nsval $SITEPKG/

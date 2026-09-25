import sys
from pathlib import Path

# Import src/elasticity_utils.py the same way the notebooks do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

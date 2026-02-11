"""
BSLVC Workflow – shared path constants.

All paths are resolved relative to the bslvc_workflow/ directory
(the parent of this lib/ package).
"""
from pathlib import Path

# ── directory layout ──
BASE_DIR   = Path(__file__).resolve().parent.parent     # bslvc_workflow/
LIB_DIR    = Path(__file__).resolve().parent             # bslvc_workflow/lib/
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

SQL_DIR         = LIB_DIR / "sql"
R_SCRIPTS_DIR   = LIB_DIR / "r_scripts"
COLUMN_NAMES_CSV = LIB_DIR / "column_names.csv"

# ── data paths ──
DB_PATH      = OUTPUT_DIR / "BSLVC_sqlite.db"
INPUT_DIR    = DATA_DIR / "input"
MAPPINGS_DIR = DATA_DIR / "mappings"

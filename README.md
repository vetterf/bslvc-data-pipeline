# BSLVC Data Pipeline

Unified data pipeline for the **Bamberg Survey of Language Contact and Change (BSLVC)** project. It covers the entire workflow from raw survey data (Excel / LimeSurvey exports) through ETL, cleansing, metadata loading, imputation, and export to SQLite, CSV, and RDS.

## Directory layout

```
bslvc-data-pipeline/
├── run_workflow.py          # single entry point
├── requirements.txt         # Python dependencies (pip freeze)
├── lib/                     # Python modules, SQL scripts, R scripts
│   ├── etl.py
│   ├── cleansing.py
│   ├── imputation.py
│   ├── limesurvey.py
│   ├── column_names.csv
│   ├── sql/                 # DDL & staging SQL
│   └── r_scripts/           # R export & imputation scripts
├── data/                    # input data & mappings  (git-ignored)
│   ├── input/               #   raw XLSX / CSV files
│   ├── mappings/             #   cleansing mapping CSVs
│   └── Feature_Overview_BSLVC.xlsx
└── output/                  # generated artefacts    (git-ignored)
    ├── BSLVC_sqlite.db
    ├── *.csv / *.rds
    └── imputation_*.txt
```

> **Note:** `data/input/`, `data/Feature_Overview_BSLVC.xlsx`, and `output/` are excluded from version control via `.gitignore` because they contain large binary data files.

## Prerequisites

- Python 3.12+
- (Optional) R with `Rscript` on PATH for RDS export and R-based imputation

## Setup

```bash
# Clone the repository
git clone https://github.com/vetterf/bslvc-data-pipeline.git
cd bslvc-data-pipeline

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Place the input data files into `data/input/` and `data/Feature_Overview_BSLVC.xlsx` before running the pipeline.

## Usage

```bash
python run_workflow.py --run <STEP> [STEP ...]
```

### Available steps

| Step          | Description                                      |
|---------------|--------------------------------------------------|
| `convert`     | Convert LimeSurvey CSV exports → XLSX            |
| `etl`         | Full pipeline (ETL → cleansing → meta → export → imputation → export) |
| `cleansing`   | Normalise / clean data (→ export)                |
| `meta`        | Load feature metadata into DB (→ export)         |
| `imputation`  | Run variety-stratified PMM imputation (→ export) |
| `export`      | Export DB views to CSV and RDS                   |

Downstream dependencies are added automatically.

### Options

| Flag                       | Description                                       |
|----------------------------|---------------------------------------------------|
| `--cleansing-mode {update,apply}` | `update` regenerates mappings; `apply` normalises data (default) |
| `--fill-empty-with-na`     | Fill empty cells with NA during cleansing          |
| `--dry-run`                | Show execution plan without running anything       |

### Examples

```bash
python run_workflow.py --run etl                             # full pipeline
python run_workflow.py --run cleansing --cleansing-mode update # regenerate mappings
python run_workflow.py --run export                          # export only
python run_workflow.py --dry-run --run etl                   # preview plan
python run_workflow.py --run convert                         # LimeSurvey → XLSX
```




# BSLVC Data Pipeline

Unified data pipeline for the **Bamberg Survey of Language Contact and Change (BSLVC)** project. It covers the entire workflow from raw survey data (Excel / LimeSurvey exports) through ETL, cleansing, metadata loading, imputation, and export to SQLite, CSV, and RDS.

## Directory layout

```
bslvc-data-pipeline/
├── run_workflow.py          # single entry point
├── requirements.txt         # Python dependencies (pip)
├── .venv/                   # Python virtual environment (git-ignored)
├── renv.lock                # R dependencies (renv lockfile)
├── .Rprofile                # auto-activates renv on Rscript calls
├── renv/                    # R virtual environment (git-ignored)
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
- R 4.x with `Rscript` on PATH (required for RDS export and R-based imputation)

## Setup

### Python environment

```bash
# Clone the repository
git clone https://github.com/vetterf/bslvc-data-pipeline.git
cd bslvc-data-pipeline

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### R environment

The project uses [renv](https://rstudio.github.io/renv/) to isolate R package dependencies. All packages (including `fabOF` from GitHub) are recorded in `renv.lock`.

```bash
# Restore the exact R package versions declared in renv.lock
Rscript -e "renv::restore()"
```

renv activates automatically via `.Rprofile` whenever `Rscript` is invoked from the project root — no manual steps are needed afterwards. The Python pipeline sets `cwd` to the project root when calling R scripts, so the environment is always picked up correctly.

> **First-time install only:** if `renv` itself is not yet installed in your system R library, run  
> `Rscript -e "install.packages('renv')"` before `renv::restore()`.

### Recreating the environments from scratch

If you need to rebuild both environments completely (e.g. after upgrading R or Python):

```bash
# ── Python ──────────────────────────────────────────────────
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ── R ───────────────────────────────────────────────────────
Rscript -e "install.packages('renv')"   # if not already installed
Rscript -e "renv::restore()"            # installs all packages from renv.lock
```

To update the R lockfile after adding or upgrading packages:

```bash
Rscript -e "renv::install('some_package')"   # install the new package
Rscript -e "renv::snapshot()"               # update renv.lock
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
| `imputation`  | Run imputation with the selected method (→ export) |
| `export`      | Export DB views to CSV and RDS                   |

Downstream dependencies are added automatically.

### Options

| Flag                                         | Description                                                                                  |
|----------------------------------------------|----------------------------------------------------------------------------------------------|
| `--update`                                   | Incremental update: add new participants from `data/input/` to the existing database. Runs cleansing, imputation, and export only for newly added participants. |
| `--update-mode {skip,overwrite}`             | How to handle participants already in the database during `--update`. `skip` (default) ignores duplicates; `overwrite` deletes and reimports them. |
| `--cleansing-mode {update,apply}`            | `update` regenerates mappings; `apply` normalises data (default)                             |
| `--imputation-method {missforest,pmm,fabof}` | Imputation method (default: `missforest`)                                                    |
| `--fill-empty-with-na`                       | Fill empty cells with NA during cleansing                                                    |
| `--dry-run`                                  | Show execution plan without running anything                                                 |

### Examples

```bash
python run_workflow.py --run etl                              # full pipeline
python run_workflow.py --run cleansing --cleansing-mode update  # regenerate mappings
python run_workflow.py --run export                           # export only
python run_workflow.py --dry-run --run etl                    # preview plan
python run_workflow.py --run convert                          # LimeSurvey → XLSX
python run_workflow.py --run imputation --imputation-method fabof  # fabOF imputation
python run_workflow.py --run imputation --imputation-method pmm    # PMM imputation
python run_workflow.py --update                               # incremental update (skip duplicates)
python run_workflow.py --update --update-mode overwrite       # incremental update (overwrite duplicates)
python run_workflow.py --update --imputation-method pmm       # incremental update with PMM
```

### Imputation methods

| Method        | Backend | Description                                                                 |
|---------------|---------|-----------------------------------------------------------------------------|
| `missforest`  | R       | Random-forest chained-equations imputation (default, best accuracy, ~10 min) |
| `pmm`         | Python  | Variety-stratified Predictive Mean Matching with cross-modality predictors (~1.5 min) |
| `fabof`       | R       | Frequency-Adjusted Borders Ordinal Forest (fabOF; Buczak 2025) — treats each variable as ordinal in a chained-equations framework with OOB convergence monitoring |

All methods impute grammar (0–5 scale) and lexical (−2 to +2 scale) data, apply per-participant missingness cutoffs, and upload the results to the SQLite database.




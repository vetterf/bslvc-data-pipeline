#!/usr/bin/env python3
"""
BSLVC Unified Workflow
======================
Single entry point for the entire BSLVC data pipeline.

Directory layout
─────────────────
  bslvc_workflow/
    run_workflow.py          ← this script
    lib/                     ← Python modules, SQL files, R scripts
    data/                    ← input xlsx, mappings
    output/                  ← SQLite DB, CSV/RDS exports

Usage
─────
  python run_workflow.py --run etl           # full pipeline
  python run_workflow.py --run cleansing     # cleansing → export
  python run_workflow.py --run export        # export only
  python run_workflow.py --run imputation    # imputation → export
  python run_workflow.py --run meta          # meta → export
  python run_workflow.py --dry-run --run etl # show plan without executing
"""

import argparse
import csv
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

# ── make bslvc_workflow/ importable ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib import DATA_DIR, DB_PATH, INPUT_DIR, OUTPUT_DIR, R_SCRIPTS_DIR


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline construction
# ═══════════════════════════════════════════════════════════════════════════════

VALID_STEPS = {"convert", "etl", "cleansing", "meta", "imputation", "export"}

# Each step maps to its full dependency chain (downstream).
STEP_CHAINS = {
    "convert":    ["convert"],
    "etl":        ["etl", "cleansing", "meta", "export", "imputation", "export"],
    "cleansing":  ["cleansing", "export"],
    "meta":       ["meta", "export"],
    "imputation": ["imputation", "export"],
    "export":     ["export"],
}


def build_pipeline(requested_steps: list[str]) -> list[str]:
    """Build an ordered execution plan from one or more requested steps.

    When a single step is requested its predefined chain is used directly.
    When multiple steps are requested their chains are merged in canonical
    order, with a final export appended when imputation is included.
    """
    if len(requested_steps) == 1:
        return list(STEP_CHAINS[requested_steps[0]])

    # Merge all chains
    active = set()
    for step in requested_steps:
        active.update(STEP_CHAINS[step])

    canonical = ["etl", "cleansing", "meta", "export", "imputation"]
    pipeline = [s for s in canonical if s in active]

    # If imputation is included, add a trailing export for re-export
    if "imputation" in active:
        pipeline.append("export")

    return pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage: CONVERT LIMESURVEY → XLSX
# ═══════════════════════════════════════════════════════════════════════════════

def run_convert():
    """Convert LimeSurvey CSV exports into XLSX files for the ETL pipeline."""
    from lib.limesurvey import convert_all_limesurvey_exports
    convert_all_limesurvey_exports()


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage: META TO DB
# ═══════════════════════════════════════════════════════════════════════════════

def run_meta():
    """Load Feature_Overview_BSLVC.xlsx (sheet 'Features') → bslvc_meta table."""
    import pandas as pd

    print()
    print("=" * 80)
    print("  STAGE: META TO DB")
    print("=" * 80)

    xlsx_path = DATA_DIR / "Feature_Overview_BSLVC.xlsx"
    if not xlsx_path.exists():
        print(f"  ⚠  {xlsx_path} not found. Skipping meta.")
        return

    print(f"  reading: {xlsx_path}")
    df = pd.read_excel(str(xlsx_path), sheet_name="Features")
    print(f"  writing table 'bslvc_meta' ({len(df)} rows) → {DB_PATH}")

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS bslvc_meta")
        conn.commit()
        df.to_sql("bslvc_meta", conn, if_exists="replace", index=False)

    print("  ✅ Meta-to-DB completed successfully")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage: EXPORT (CSV + RDS)
# ═══════════════════════════════════════════════════════════════════════════════

EXPORT_VIEWS = [
    "Informants",
    "BSLVC_ALL",
    "BSLVC_GRAMMAR",
    "BSLVC_SPOKEN",
    "BSLVC_WRITTEN",
    "BSLVC_LEXICAL",
]


def run_export():
    """Export all views to semicolon-separated CSV and RDS (via Rscript)."""
    import pandas as pd

    print()
    print("=" * 80)
    print("  STAGE: DB EXPORT TO CSV / RDS")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"  ⚠  Database not found at {DB_PATH}. Skipping export.")
        return

    # ── CSV export ──────────────────────────────────────────────────────────
    print(f"  exporting CSV from {DB_PATH}")
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            for view in EXPORT_VIEWS:
                df = pd.read_sql_query(f"SELECT * FROM {view}", conn)
                out = OUTPUT_DIR / f"{view}.csv"
                df.to_csv(
                    str(out), index=False, header=True, sep=";",
                    encoding="utf-8", quoting=csv.QUOTE_ALL,
                )
                print(f"    {view}.csv  ({len(df)} rows)")
    except sqlite3.Error as e:
        print(f"  ❌ CSV export failed: {e}")
        return

    # ── RDS export ──────────────────────────────────────────────────────────
    r_script = R_SCRIPTS_DIR / "BSLVC_export.R"
    if not r_script.exists():
        print(f"  ⚠  R export script not found at {r_script}. Skipping RDS.")
        return

    print("  exporting RDS via Rscript …")
    try:
        result = subprocess.run(
            ["Rscript", str(r_script), str(OUTPUT_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        else:
            print(f"  ❌ Rscript error:\n{result.stderr}")
            return
    except FileNotFoundError:
        print("  ⚠  Rscript not found in PATH. RDS export skipped.")

    print("  ✅ Export completed successfully")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage: IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_imputation(method: str = "missforest"):
    """Run imputation with the specified method.

    Parameters
    ----------
    method : str
        ``'missforest'`` (default) — R missForest (best accuracy, slower).
        ``'pmm'`` — Python variety-stratified PMM (faster, good accuracy).
        ``'fabof'`` — R fabOF ordinal chained-forest (proper ordinal treatment).
    """
    from lib.imputation import run_imputation as _run_imp
    _run_imp(method=method)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def execute_pipeline(
    pipeline,
    *,
    cleansing_mode="apply",
    fill_empty_with_na=False,
    imputation_method="missforest",
    dry_run=False,
):
    """Execute each stage in *pipeline* sequentially."""
    print()
    print("━" * 80)
    print(f"  BSLVC Workflow – execution plan: {' → '.join(pipeline)}")
    if "imputation" in pipeline:
        print(f"  Imputation method: {imputation_method}")
    print("━" * 80)

    if dry_run:
        print()
        print("  (dry run – no stages will be executed)")
        return

    # Lazy imports – heavy dependencies (pandas, flair, …) are only loaded
    # when a stage actually runs, so --dry-run / --help work without them.
    from lib.etl import run_etl
    from lib.cleansing import run_cleansing

    dispatch = {
        "convert":    run_convert,
        "etl":        run_etl,
        "meta":       run_meta,
        "export":     run_export,
    }

    for stage in pipeline:
        if stage == "cleansing":
            run_cleansing(
                mode=cleansing_mode,
                fill_empty_with_na=fill_empty_with_na,
            )
        elif stage == "imputation":
            run_imputation(method=imputation_method)
        else:
            dispatch[stage]()

    print()
    print("━" * 80)
    print("  ✅ All stages finished.")
    print("━" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BSLVC Unified Workflow – single entry point for the data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Pipeline dependency chains
            ──────────────────────────
              etl        → cleansing → meta → export → imputation → export
              cleansing  → export
              meta       → export
              imputation → export
              export     → (standalone)
              convert    → (standalone – output requires manual review)

            Examples
            ────────
              %(prog)s --run convert                        Convert LimeSurvey CSVs
              %(prog)s --run etl                            Full pipeline
              %(prog)s --run cleansing --cleansing-mode update
              %(prog)s --run export                         Export only
              %(prog)s --dry-run --run etl                  Show plan
        """),
    )

    parser.add_argument(
        "--run", nargs="+", required=True,
        choices=sorted(VALID_STEPS),
        metavar="STEP",
        help="Stage(s) to run.  Downstream dependencies are added automatically.  "
             f"Choices: {', '.join(sorted(VALID_STEPS))}",
    )
    parser.add_argument(
        "--cleansing-mode", choices=["update", "apply"], default="apply",
        help="Cleansing mode: 'update' regenerates mappings, "
             "'apply' normalises data (default: apply).",
    )
    parser.add_argument(
        "--fill-empty-with-na", action="store_true",
        help="When applying cleansing, fill empty cells with NA.",
    )
    parser.add_argument(
        "--imputation-method",
        choices=["missforest", "pmm", "fabof"],
        default="missforest",
        help="Imputation method: 'missforest' (default, R missForest — best "
             "accuracy, ~10 min), 'pmm' (Python variety-stratified PMM — "
             "faster, ~1.5 min), or 'fabof' (R fabOF ordinal chained-forest "
             "— proper ordinal treatment).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the execution plan without running any stage.",
    )

    args = parser.parse_args()

    pipeline = build_pipeline(args.run)
    execute_pipeline(
        pipeline,
        cleansing_mode=args.cleansing_mode,
        fill_empty_with_na=args.fill_empty_with_na,
        imputation_method=args.imputation_method,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

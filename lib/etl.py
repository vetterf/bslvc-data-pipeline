"""
ETL Module – Extract, Transform, Load
======================================
Reads Excel questionnaire files from data/input/, cleans them,
and loads into the SQLite database at data/BSLVC_sqlite.db.

Ported from bslvc_ETL/etl.py + bslvc_ETL/main.py.
"""

import pandas as pd
import re
import sqlite3
import glob
import os
from pathlib import Path

from lib import SQL_DIR, COLUMN_NAMES_CSV, DATA_DIR, DB_PATH, INPUT_DIR, OUTPUT_DIR


# ═══════════════════════════════════════════════════════════════════════════
#  SQL helpers
# ═══════════════════════════════════════════════════════════════════════════

def read_sql_files(folder: Path = SQL_DIR) -> dict:
    """Read all .sql files from *folder* into a dict keyed by stem name."""
    sql_queries = {}
    for fpath in sorted(folder.glob("*.sql")):
        sql_queries[fpath.stem] = fpath.read_text()
    return sql_queries


# ═══════════════════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_from_csv(file_to_process):
    return pd.read_csv(file_to_process, sep=";")


def extract_from_xlsx(file_to_process):
    return pd.read_excel(file_to_process, dtype=str, engine="openpyxl")


# ═══════════════════════════════════════════════════════════════════════════
#  Cleaning / Transformation
# ═══════════════════════════════════════════════════════════════════════════

def drop_na_in_first_column(df):
    first_column = df.columns[0]
    return df.dropna(subset=[first_column])


def clean(data):
    """Clean and reshape a single questionnaire DataFrame."""
    colmap = pd.read_csv(str(COLUMN_NAMES_CSV), header=None, sep=";")
    data = drop_na_in_first_column(data).copy()

    # Handle duplicate "Comments" rows
    Comments_rows = data[data.iloc[:, 0] == "Comments"]
    if len(Comments_rows) > 0:
        data.loc[Comments_rows.index, data.columns[0]] = [
            "CommentsGeneral",
            "CommentsLexical",
        ]

    data = data.transpose()
    data.columns = data.iloc[0,]
    data = data.iloc[1:,]
    data = data.assign(InformantID=data.index)
    data = data.dropna(subset=["InformantID"])

    collectionCountry_match = re.search(r"^[A-Za-z]+", data["InformantID"].iloc[0])
    collectionCountry = collectionCountry_match[0] if collectionCountry_match else ""

    collectionYear_match = re.search(r"^[A-Za-z]+(\d+)", data["InformantID"].iloc[0])
    if collectionYear_match:
        collectionYear = "20" + collectionYear_match.group(1)
    else:
        # Fallback: try to find any digits in the ID
        digits_match = re.search(r"\d+", data["InformantID"].iloc[0])
        collectionYear = "20" + digits_match[0] if digits_match else ""

    data["CountryCollection"] = collectionCountry

    data["Date"] = ""
    data["Year"] = collectionYear

    data.columns = data.columns.str.replace(r"(\n|\r|\t)", "", regex=True)

    # Merge timeline comments and general comments
    data["CommentsTimeline"] = (
        data["Timeline Comments"]
        .fillna("")
        .astype(str)
        .combine(
            data["CommentsGeneral"].fillna("").astype(str),
            lambda a, b: ((a or "") + (b or "")) or None,
            None,
        )
    )

    data["GrammarSpokenFillingInFor"] = ""
    data["GrammarWrittenFillingInFor"] = ""

    # Drop unnecessary columns
    dropcols = [
        "CommentsGeneral",
        "Lexical sets",
        "Grammar section 1",
        "Grammar section 2",
        "Timeline Comments",
        "Lexical Sets",
    ]
    for col in dropcols:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Handle timeline columns with fuzzy matching
    timelineCols = [
        "Years lived outside of home country",
        "Years lived in home country",
        "Years lived in other English-speaking countries",
    ]
    if len([data.columns.get_loc(c) for c in timelineCols if c in data]) < 3:
        patterns = [
            (r"Years lived in (other|English).+", "YearsLivedOtherEnglish"),
            (r"Years lived in(?!\s*(other|English)).+", "YearsLivedInside"),
            (r"Years lived outside.+", "YearsLivedOutside"),
        ]
        for pattern, replacement in patterns:
            matching_columns = data.filter(regex=pattern).columns
            data = data.rename(columns={matching_columns[0]: replacement})
        if len([data.columns.get_loc(c) for c in timelineCols if c in data]) < 3:
            print("Skipping questionnaire, double check column names")
            return pd.DataFrame()

    # Rename using column map
    for row in colmap.iterrows():
        if row[1][1] in data.columns:
            data.rename(columns={row[1][1]: row[1][0]}, inplace=True)

    data = data.drop_duplicates(subset=["InformantID"])
    data = data.replace("NA", "")
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    data = data.round(2)

    # Remove secondary-school identifying information (name / place)
    for col in ("NameSchool",):
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    return data


# ═══════════════════════════════════════════════════════════════════════════
#  Staging & Loading
# ═══════════════════════════════════════════════════════════════════════════

def stage(data, conn, createStagingQuery, clear=True):
    cur = conn.cursor()
    if clear:
        cur.execute("DROP TABLE IF EXISTS bslvc_staging_table")
        print("  staging table dropped")
        cur.executescript(createStagingQuery)
        conn.commit()
        print("  recreated staging table")

    print("  loading data to staging table")
    data.to_sql("bslvc_staging_table", con=conn, if_exists="append", index=False)
    print("  data loaded to staging table")


def etl(query, file, conn):
    """Extract one file, clean, stage, and load into final tables."""
    print(f"  reading: {file}")
    if file.endswith(".xls") or file.endswith(".xlsx"):
        data = extract_from_xlsx(file)
    else:
        data = extract_from_csv(file)
    print("  file read successfully")

    data = clean(data)
    if data.empty:
        print("  data is empty – skipping")
        return

    cur = conn.cursor()
    try:
        stage(data, conn, clear=True, createStagingQuery=query["01_create_staging_table_bslvc"])
        print("  data loaded into staging table")

        cur.executescript(query["04_staging_informant_to_table"])
        cur.executescript(query["05_staging_lexical_to_table"])
        cur.executescript(query["06_staging_grammar_spoken_to_table"])
        cur.executescript(query["07_staging_grammar_written_to_table"])

        conn.commit()
        print("  data loaded from staging into final tables")
    except sqlite3.Error as error:
        print(f"  error: {error}")
        conn.rollback()
        print("  rolled back transaction")


def create_tables(conn, queries):
    cur = conn.cursor()
    try:
        cur.executescript(queries["00_delete_all_tables"])
        cur.executescript(queries["02_create_tables_bslvc"])
        cur.executescript(queries["03_CreateViews"])
        if "09_create_lexical_columns_table" in queries:
            cur.executescript(queries["09_create_lexical_columns_table"])
        conn.commit()
    except sqlite3.Error as error:
        print(f"  error: {error}")
        conn.rollback()
        print("  rolled back transaction")


def etl_process(sql_queries, files, conn):
    """Full ETL: create tables, load Lexical_Columns, process each file, write metadata."""
    create_tables(conn, sql_queries)

    # Load Lexical_Columns.csv if present
    lexical_columns_path = DATA_DIR / "Lexical_Columns.csv"
    try:
        if lexical_columns_path.exists():
            print(f"  loading lexical columns from: {lexical_columns_path}")
            lexical_df = pd.read_csv(str(lexical_columns_path), sep=";")
            lexical_df.to_sql("LexicalColumns", conn, if_exists="replace", index=False)
            print(f"  loaded {len(lexical_df)} lexical column entries")
    except Exception as error:
        print(f"  warning: could not load Lexical_Columns.csv: {error}")

    for file in files:
        etl(sql_queries, file, conn)

    _trim_and_update_metadata(conn, sql_queries, len(files), "created")


def _trim_and_update_metadata(conn, sql_queries, n_files, verb):
    """Trim whitespace from Informants text columns and update metadata."""
    # Strip trailing whitespace from all text columns in Informants
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(Informants)")
        text_cols = [row[1] for row in cur.fetchall() if row[2].upper() in ('TEXT', '')]
        for col in text_cols:
            cur.execute(f'UPDATE Informants SET "{col}" = TRIM("{col}") WHERE "{col}" IS NOT NULL')
        conn.commit()
        print("  trimmed whitespace from Informants text columns")
    except sqlite3.Error as error:
        print(f"  warning: could not trim Informants columns: {error}")

    # Create/update metadata table
    try:
        cur = conn.cursor()
        if "08_create_metadata_table" in sql_queries:
            cur.executescript(sql_queries["08_create_metadata_table"])
            cur.execute("SELECT COUNT(*) FROM Informants")
            total_informants = cur.fetchone()[0]
            cur.execute(
                "UPDATE DatabaseMetadata SET TotalInformants = ?, Notes = ? WHERE ID = 1",
                (total_informants, f"Database {verb} with {n_files} data files"),
            )
            conn.commit()
            print(f"  metadata updated: {total_informants} informants processed")
    except sqlite3.Error as error:
        print(f"  warning: could not update metadata: {error}")


def etl_update_process(sql_queries, files, conn, *, update_mode="skip"):
    """Incremental ETL: add new participants to existing tables.

    *update_mode* controls duplicate handling:
    ``'skip'`` – ignore participants already in the DB.
    ``'overwrite'`` – delete existing data for those participants and reimport.

    Returns a list of InformantIDs that were (re)imported.
    """
    cur = conn.cursor()

    # Get existing InformantIDs
    cur.execute("SELECT InformantID FROM Informants")
    existing_ids = {row[0] for row in cur.fetchall()}
    print(f"  existing participants in database: {len(existing_ids)}")

    new_ids_all = []

    # Tables that hold per-participant data (order: children first for FK safety)
    _PARTICIPANT_TABLES = [
        "LexicalItemsImputed", "SpokenItemsImputed", "WrittenItemsImputed",
        "LexicalItems", "SpokenItems", "WrittenItems",
        "Informants",
    ]

    for file in files:
        print(f"  reading: {file}")
        if file.endswith(".xls") or file.endswith(".xlsx"):
            data = extract_from_xlsx(file)
        else:
            data = extract_from_csv(file)
        print("  file read successfully")

        data = clean(data)
        if data.empty:
            print("  data is empty – skipping")
            continue

        # Filter out participants that already exist in the database
        duplicate_mask = data["InformantID"].isin(existing_ids)
        n_duplicates = duplicate_mask.sum()

        if update_mode == "overwrite" and n_duplicates > 0:
            # Delete existing data for duplicates so they get reimported
            dup_ids = data.loc[duplicate_mask, "InformantID"].tolist()
            placeholders = ",".join("?" * len(dup_ids))
            for tbl in _PARTICIPANT_TABLES:
                try:
                    conn.execute(
                        f"DELETE FROM {tbl} WHERE InformantID IN ({placeholders})",
                        dup_ids,
                    )
                except sqlite3.OperationalError:
                    pass  # table may not exist yet (e.g. imputed tables)
            conn.commit()
            print(f"  {n_duplicates} existing participants deleted for reimport")
            existing_ids -= set(dup_ids)
            # keep all rows – duplicates will be reimported
            new_ids = data["InformantID"].tolist()
            skipped = 0
        else:
            # skip mode: drop duplicates
            data = data[~duplicate_mask].copy()
            skipped = n_duplicates

            if data.empty:
                print(f"  all {skipped} participants already in database – skipping")
                continue

            new_ids = data["InformantID"].tolist()
        new_ids_all.extend(new_ids)
        if skipped:
            print(f"  {len(new_ids)} new participants, {skipped} duplicates skipped")
        else:
            print(f"  {len(new_ids)} participants to import")

        try:
            stage(data, conn, clear=True,
                  createStagingQuery=sql_queries["01_create_staging_table_bslvc"])
            print("  data loaded into staging table")

            cur.executescript(sql_queries["04_staging_informant_to_table"])
            cur.executescript(sql_queries["05_staging_lexical_to_table"])
            cur.executescript(sql_queries["06_staging_grammar_spoken_to_table"])
            cur.executescript(sql_queries["07_staging_grammar_written_to_table"])

            conn.commit()
            print("  new data loaded from staging into final tables")

            # Add new IDs to the existing set so subsequent files also skip them
            existing_ids.update(new_ids)
        except sqlite3.Error as error:
            print(f"  error: {error}")
            conn.rollback()
            print("  rolled back transaction")

    if new_ids_all:
        _trim_and_update_metadata(conn, sql_queries, len(files), "updated")
    else:
        print("  no new participants found in any file")

    return new_ids_all


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_etl():
    """Run the full ETL process: read xlsx from data/input/, load into SQLite."""
    print()
    print("=" * 80)
    print("  STAGE: ETL – Loading Excel data into SQLite")
    print("=" * 80)

    sql_queries = read_sql_files()
    filelist = sorted(glob.glob(str(INPUT_DIR / "*.xlsx")))

    if not filelist:
        print(f"  ⚠  No .xlsx files found in {INPUT_DIR}. Skipping ETL.")
        return

    print(f"  found {len(filelist)} data file(s)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  database: {DB_PATH}")

    with sqlite3.connect(str(DB_PATH)) as conn:
        etl_process(sql_queries, filelist, conn)

    print("  ✅ ETL completed successfully")


def run_etl_update(*, update_mode="skip"):
    """Incremental ETL: add new participants from data/input/ to existing SQLite DB.

    Returns a list of newly added InformantIDs.
    """
    print()
    print("=" * 80)
    print("  STAGE: ETL UPDATE – Adding new participants to existing database")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"  ⚠  Database not found at {DB_PATH}.")
        print("     Run the full ETL first (--run etl) to create the database.")
        return []

    sql_queries = read_sql_files()
    filelist = sorted(glob.glob(str(INPUT_DIR / "*.xlsx")))

    if not filelist:
        print(f"  ⚠  No .xlsx files found in {INPUT_DIR}. Skipping ETL update.")
        return []

    print(f"  found {len(filelist)} data file(s)")
    print(f"  database: {DB_PATH}")

    with sqlite3.connect(str(DB_PATH)) as conn:
        new_ids = etl_update_process(sql_queries, filelist, conn, update_mode=update_mode)

    print(f"  ✅ ETL update completed: {len(new_ids)} new participants added")
    return new_ids

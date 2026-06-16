"""
LimeSurvey Conversion Module
=============================
Converts LimeSurvey CSV exports (``results-surveyNNN.csv``) into the
transposed XLSX format expected by the ETL pipeline.

Workflow
--------
1. Place the CSV export from LimeSurvey in ``data/conversion/input/``.
2. Run ``python run_workflow.py --run convert``.
3. **First run** (no mapping file yet): a stub mapping file is created at
   ``data/conversion/<surveyNNN>_mapping.csv`` with auto-detected column
   assignments. Review / adjust that file, then re-run.
4. **Subsequent runs**: the mapping file drives the conversion. The resulting
   XLSX is written to ``data/conversion/output/`` and the CSV is moved to
   ``data/conversion/input/Done/<timestamp>/``.

Mapping file format (semicolon-separated)
------------------------------------------
  question_code ; csv_column ; target_row ; transform

``question_code``
    Code extracted from the LimeSurvey column header (e.g. ``PI01``).
``csv_column``
    Full original column header – for reference only.
``target_row``
    Row label in the output XLSX as defined by the template, **or** one
    of the special values below.  Leave empty to mark the column as
    unmapped; the corresponding template rows will be filled with ``ND``.
``transform``
    How the raw CSV value is converted (empty = plain copy):

    ``other``              – append as free-text to the preceding base field
                             that shares the same ``target_row``.
    ``lang_checkbox:LABEL``– yes/no checkbox; if "Yes", contribute LABEL to
                             the semicolon-joined list for ``target_row``.
    ``lang_other``         – free-text, appended to the lang list.
    ``qual_checkbox:LABEL``– yes/no checkbox for a qualification category.
    ``qual_other``         – free-text qualification (appended).
    ``grammar_value``      – map AO01-AO06 (or legacy No-one…Everyone) → 0-5.
    ``lexical_value``      – map LIA1/LIA2/LIA3/LIA4/LIA5/LIA6 → -2/-1/0/1/2/NX;
                             if the target row label ends with `` -``, the
                             numeric result is multiplied by -1 (reversed item).
                             Empty answers are left as empty (NA in the DB).
    ``fixed:<value>``      – always output the given literal value, regardless
                             of the CSV content (e.g. ``fixed:1`` for Signature).
    ``skip``               – ignore this column entirely.

Derived rows (no mapping entry needed)
---------------------------------------
``Ratio``
    Computed as *Years lived outside of home country / Age*.
    If *Years lived outside* is empty → ``1``.
    If *Age* is missing or zero, or on any error → ``NA``.
"""

import re
import shutil
import pandas as pd
from pathlib import Path

from lib import DATA_DIR, INPUT_DIR, MAPPINGS_DIR


# ── Directory layout ─────────────────────────────────────────────────────────
CONVERSION_DIR = DATA_DIR / "conversion"
CONVERSION_INPUT_DIR = CONVERSION_DIR / "input"
CONVERSION_OUTPUT_DIR = CONVERSION_DIR / "output"
TEMPLATE_XLSX = MAPPINGS_DIR / "template.xlsx"

# Legacy export directory (kept for backwards compatibility)
LIMESURVEY_DIR = INPUT_DIR / "Limesurvey_export"

# LimeSurvey lexical answer codes → numeric scale
_LEXICAL_VALUE_MAP = {
    "LIA1": "-2",
    "LIA2": "-1",
    "LIA3": "0",
    "LIA4": "1",
    "LIA5": "2",
    "LIA6": "NX",
}

# Grammar answer text → numeric scale
_GRAMMAR_VALUE_MAP = {
    # LimeSurvey answer codes
    "AO01": "5",
    "AO02": "4",
    "AO03": "3",
    "AO04": "2",
    "AO05": "1",
    "AO06": "0",
    # Legacy text labels (kept for backwards compatibility)
    "No-one": "0",
    "Few": "1",
    "Some": "2",
    "Many": "3",
    "Most": "4",
    "Everyone": "5",
}

# ── Auto-detect for R syntax file exports (underscore column notation) ────
# Column codes here match what the R syntax file assigns via names(data)[N].
_AUTO_DETECT_R: list[tuple[str, str, str]] = [
    ("id",                    "",                                             "skip"),
    ("submitdate",            "",                                             "skip"),
    ("lastpage",              "",                                             "skip"),
    ("startlanguage",         "",                                             "skip"),
    ("seed",                  "",                                             "skip"),
    ("startdate",             "",                                             "skip"),
    ("datestamp",             "",                                             "skip"),
    ("CF01",                  "",                                             "skip"),
    ("PI01",                  "Age",                                          ""),
    ("PI02",                  "Gender",                                       ""),
    ("PI02_other",            "Gender",                                       "other"),
    ("PI03",                  "Nationality",                                  ""),
    ("PI03_other",            "Nationality",                                  "other"),
    ("PI04",                  "Ethnic self-identification",                   ""),
    ("PI05",                  "Country or region you identify with",          ""),
    ("PI06_SQ001SQ001",       "Languages used at home while  growing up",    "lang_checkbox:German"),
    ("PI06_SQ002",            "Languages used at home while  growing up",    "lang_checkbox:English"),
    ("PI06_SQ003",            "Languages used at home while  growing up",    "lang_checkbox:French"),
    ("PI06_SQ004",            "Languages used at home while  growing up",    "lang_checkbox:Italian"),
    ("PI06_SQ005",            "Languages used at home while  growing up",    "lang_checkbox:Spanish"),
    ("PI06_SQ006",            "Languages used at home while  growing up",    "lang_checkbox:Hungarian"),
    ("PI06_SQ007",            "Languages used at home while  growing up",    "lang_checkbox:Polish"),
    ("PI06_SQ008",            "Languages used at home while  growing up",    "lang_checkbox:Czech"),
    ("PI06_SQ009",            "Languages used at home while  growing up",    "lang_checkbox:Irish"),
    ("PI06_other",            "Languages used at home while  growing up",    "lang_other"),
    ("PI07m_SQ001SQ001",      "Native Lg. Mother",                           "lang_checkbox:German"),
    ("PI07m_SQ002",           "Native Lg. Mother",                           "lang_checkbox:English"),
    ("PI07m_SQ003",           "Native Lg. Mother",                           "lang_checkbox:French"),
    ("PI07m_SQ004",           "Native Lg. Mother",                           "lang_checkbox:Italian"),
    ("PI07m_SQ005",           "Native Lg. Mother",                           "lang_checkbox:Spanish"),
    ("PI07m_SQ006",           "Native Lg. Mother",                           "lang_checkbox:Hungarian"),
    ("PI07m_SQ007",           "Native Lg. Mother",                           "lang_checkbox:Polish"),
    ("PI07m_SQ008",           "Native Lg. Mother",                           "lang_checkbox:Czech"),
    ("PI07m_SQ009",           "Native Lg. Mother",                           "lang_checkbox:Irish"),
    ("PI07m_other",           "Native Lg. Mother",                           "lang_other"),
    ("PI07f_SQ001SQ001",      "Native Lg. Father",                           "lang_checkbox:German"),
    ("PI07f_SQ002",           "Native Lg. Father",                           "lang_checkbox:English"),
    ("PI07f_SQ003",           "Native Lg. Father",                           "lang_checkbox:French"),
    ("PI07f_SQ004",           "Native Lg. Father",                           "lang_checkbox:Italian"),
    ("PI07f_SQ005",           "Native Lg. Father",                           "lang_checkbox:Spanish"),
    ("PI07f_SQ006",           "Native Lg. Father",                           "lang_checkbox:Hungarian"),
    ("PI07f_SQ007",           "Native Lg. Father",                           "lang_checkbox:Polish"),
    ("PI07f_SQ008",           "Native Lg. Father",                           "lang_checkbox:Czech"),
    ("PI07f_SQ009",           "Native Lg. Father",                           "lang_checkbox:Irish"),
    ("PI07f_other",           "Native Lg. Father",                           "lang_other"),
    ("PI08",                  "Years lived outside of home country",         ""),
    ("PI08Copy",              "Years lived in home country",                 ""),
    ("PI09",                  "",                                             "skip"),
    ("PI10",                  "Timeline Comments",                            ""),
    ("PI11",                  "Years lived in other English-speaking countries", ""),
    ("PI12",                  "",                                             "skip"),
    ("PI12_other",            "",                                             "skip"),
    ("HC2",                   "",                                             "skip"),
    ("EP01",                  "Primary school",                               ""),
    ("EP01_other",            "Primary school",                              "other"),
    ("EP02",                  "Secondary school",                             ""),
    ("EP02_other",            "Secondary school",                            "other"),
    ("EP03",                  "Name and Place of high school",                ""),
    ("EP04_SQ001",            "Qualifications",                              "qual_checkbox:Apprenticeship"),
    ("EP04_SQ002",            "Qualifications",                              "qual_checkbox:Vocational classes"),
    ("EP04_SQ007",            "Qualifications",                              "qual_checkbox:General secondary education"),
    ("EP04_SQ003",            "Qualifications",                              "qual_checkbox:Bachelor"),
    ("EP04_SQ004",            "Qualifications",                              "qual_checkbox:Master's"),
    ("EP04_SQ005",            "Qualifications",                              "qual_checkbox:PhD"),
    ("EP04_SQ006",            "Qualifications",                              "qual_checkbox:Other"),
    ("EP04_other",            "Qualifications",                              "qual_other"),
    ("EP05",                  "Current occupation",                           ""),
    ("EP05_other",            "Current occupation",                          "other"),
    ("EP06",                  "Qualification mother",                        ""),
    ("EP06_other",            "Qualification mother",                        "other"),
    ("EP07",                  "Qualification father",                        ""),
    ("EP07_other",            "Qualification father",                        "other"),
    ("EP08",                  "Qualification partner",                       ""),
    ("EP08_other",            "Qualification partner",                       "other"),
    ("EP09_SQ002",            "Occupation mother",                           ""),
    ("EP09_SQ004",            "Occupation father",                           ""),
    ("EP09_SQ006",            "Occupation partner",                          ""),
    ("LI1C",                  "Comments",                                    "qual_other"),
    ("L2C",                   "Comments",                                    "qual_other"),
    ("L3C",                   "Comments",                                    "qual_other"),
    ("L4C",                   "Comments",                                    "qual_other"),
    ("G24Q42",                "",                                             "skip"),
    # Fixed values (no CSV column needed)
    ("_signature",            "Signature",                                   "fixed:1"),
    # Lexical items  (LIA1-LIA6 codes are restored by run_limesurvey_syntax.R)
    ("LI01_SQ001",            "a drop in the ocean -",                       "lexical_value"),
    ("LI01_SQ002",            "a tap",                                       "lexical_value"),
    ("LI01_SQ003",            "aluminium",                                   "lexical_value"),
    ("LI01_SQ004",            "anticlockwise -",                             "lexical_value"),
    ("LI01_SQ005",            "aubergine",                                   "lexical_value"),
    ("LI01_SQ006",            "autumn",                                      "lexical_value"),
    ("LI01_SQ007",            "backwards",                                   "lexical_value"),
    ("LI01_SQ008",            "bicentenary -",                               "lexical_value"),
    ("LI01_SQ009",            "biscuit",                                     "lexical_value"),
    ("LI01_SQ010",            "bookings -",                                  "lexical_value"),
    ("LI01_SQ011",            "boot",                                        "lexical_value"),
    ("LI01_SQ012",            "car park -",                                  "lexical_value"),
    ("LI01_SQ013",            "centre",                                      "lexical_value"),
    ("LI01_SQ014",            "chemist's -",                                 "lexical_value"),
    ("LI01_SQ015",            "ill -",                                       "lexical_value"),
    ("LI01_SQ016",            "potato chips",                                "lexical_value"),
    ("LI01_SQ017",            "chips",                                       "lexical_value"),
    ("LI02_SQ001",            "cinema -",                                    "lexical_value"),
    ("LI02_SQ002",            "colour",                                      "lexical_value"),
    ("LI02_SQ003",            "cupboard -",                                  "lexical_value"),
    ("LI02_SQ004",            "driving licence",                             "lexical_value"),
    ("LI02_SQ005",            "dummy -",                                     "lexical_value"),
    ("LI02_SQ006",            "dustbin",                                     "lexical_value"),
    ("LI02_SQ007",            "fish fingers -",                              "lexical_value"),
    ("LI02_SQ008",            "football",                                    "lexical_value"),
    ("LI02_SQ009",            "forwards -",                                  "lexical_value"),
    ("LI02_SQ010",            "globalisation",                               "lexical_value"),
    ("LI02_SQ011",            "glocalisation -",                             "lexical_value"),
    ("LI02_SQ012",            "holiday",                                     "lexical_value"),
    ("LI02_SQ013",            "liberalisation",                              "lexical_value"),
    ("LI02_SQ014",            "jacket potato",                               "lexical_value"),
    ("LI02_SQ015",            "laund(e)rette -",                             "lexical_value"),
    ("LI02_SQ016",            "potato crisps -",                             "lexical_value"),
    ("LI02_SQ017",            "crisps -",                                    "lexical_value"),
    ("LI02_SQ018",            "rubbish",                                     "lexical_value"),
    ("LI03_SQ001",            "to licence -",                                "lexical_value"),
    ("LI03_SQ002",            "lift",                                        "lexical_value"),
    ("LI03_SQ003",            "localisation -",                              "lexical_value"),
    ("LI03_SQ004",            "lorry",                                       "lexical_value"),
    ("LI03_SQ005",            "maths -",                                     "lexical_value"),
    ("LI03_SQ006",            "mobile phone",                                "lexical_value"),
    ("LI03_SQ007",            "modernisation -",                             "lexical_value"),
    ("LI03_SQ008",            "nappies",                                     "lexical_value"),
    ("LI03_SQ009",            "organisation -",                              "lexical_value"),
    ("LI03_SQ010",            "parcel",                                      "lexical_value"),
    ("LI03_SQ011",            "pavement -",                                  "lexical_value"),
    ("LI03_SQ012",            "petrol",                                      "lexical_value"),
    ("LI03_SQ013",            "petrol station -",                            "lexical_value"),
    ("LI03_SQ014",            "postman",                                     "lexical_value"),
    ("LI03_SQ015",            "pushchair -",                                 "lexical_value"),
    ("LI03_SQ016",            "railway",                                     "lexical_value"),
    ("LI03_SQ017",            "realisation -",                               "lexical_value"),
    ("LI04_SQ001",            "roundabout",                                  "lexical_value"),
    ("LI04_SQ002",            "rubber -",                                    "lexical_value"),
    ("LI04_SQ003",            "shopping trolley -",                          "lexical_value"),
    ("LI04_SQ004",            "sport",                                       "lexical_value"),
    ("LI04_SQ005",            "storm in a teacup -",                         "lexical_value"),
    ("LI04_SQ006",            "subway",                                      "lexical_value"),
    ("LI04_SQ007",            "to let -",                                    "lexical_value"),
    ("LI04_SQ008",            "torch",                                       "lexical_value"),
    ("LI04_SQ009",            "touch wood -",                                "lexical_value"),
    ("LI04_SQ010",            "trainers",                                    "lexical_value"),
    ("LI04_SQ011",            "whilst -",                                    "lexical_value"),
    ("LI04_SQ012",            "windscreen",                                  "lexical_value"),
    ("LI04_SQ013",            "a book about  chemistry -",                   "lexical_value"),
    ("LI04_SQ014",            "compare X to Y -",                            "lexical_value"),
    ("LI04_SQ015",            "typical of -",                                "lexical_value"),
    ("LI04_SQ016",            "Anyway",                                      "lexical_value"),
    # Grammar spoken: SSA_SSA1 … SSF_SSF23
    *[(f"SS{L}_SS{L}{n}", f"{L}{n}", "grammar_value")
      for L in "ABCDEF" for n in range(1, 24)],
    # Grammar written: WSG_WSG1 … WSN_WSN25
    *[(f"WSG_WSG{n}", f"G{n}", "grammar_value") for n in range(1, 27)],
    *[(f"WSH_WSH{n}", f"H{n}", "grammar_value") for n in range(1, 27)],
    *[(f"WSI_WSI{n}", f"I{n}", "grammar_value") for n in range(1, 27)],
    *[(f"WSJ_WSJ{n}", f"J{n}", "grammar_value") for n in range(1, 27)],
    *[(f"WSK_WSK{n}", f"K{n}", "grammar_value") for n in range(1, 4)],
    ("WSK_WSK4a",             "K4a",                                         "grammar_value"),
    ("WSK_WSK4b",             "K4b",                                         "grammar_value"),
    *[(f"WSK_WSK{n}", f"K{n}", "grammar_value") for n in range(5, 27)],
    *[(f"WSL_WSL{n}", f"L{n}", "grammar_value") for n in range(1, 27)],
    *[(f"WSM_WSM{n}", f"M{n}", "grammar_value") for n in range(1, 26)],
    *[(f"WSN_WSN{n}", f"N{n}", "grammar_value") for n in range(1, 26)],
]

_AUTO_DETECT_R_MAP: dict[str, tuple[str, str]] = {
    code: (target, transform) for code, target, transform in _AUTO_DETECT_R
}

# R script that extracts the labeled / LIA-restored intermediate CSV
_R_SYNTAX_RUNNER = (
    Path(__file__).parent / "r_scripts" / "run_limesurvey_syntax.R"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _survey_r_id_from_path(csv_path: Path) -> str:
    """Extract 'surveyNNN_r' from 'survey_998353_R_data_file.csv'."""
    m = re.search(r'survey_?(\d+)', csv_path.stem, re.IGNORECASE)
    return f"survey{m.group(1)}_r" if m else f"{csv_path.stem}_r"


def _r_syntax_file_for(data_csv: Path) -> Path | None:
    """Find the R syntax file that matches a given R data CSV.

    Expects the two files to live in the same directory and share the same
    survey number, e.g.  ``survey_998353_R_data_file.csv``  →
    ``survey_998353_R_syntax_file.R``.
    """
    m = re.search(r'survey_?(\d+)', data_csv.stem, re.IGNORECASE)
    if not m:
        return None
    candidate = data_csv.parent / f"survey_{m.group(1)}_R_syntax_file.R"
    return candidate if candidate.exists() else None


def _mapping_path(survey_id: str) -> Path:
    return CONVERSION_DIR / f"{survey_id}_mapping.csv"


def _load_template_rows() -> list[str]:
    """Read all row labels from the template XLSX (first column)."""
    tpl = pd.read_excel(str(TEMPLATE_XLSX), header=None, dtype=str)
    return [str(v) if str(v) != "nan" else "" for v in tpl.iloc[:, 0]]


def _clean_val(val) -> str:
    """Normalize a raw cell value to a plain string (empty if NaN/nan)."""
    s = str(val).strip() if pd.notna(val) else ""
    return "" if s.lower() == "nan" else s


# ═══════════════════════════════════════════════════════════════════════════
#  Stub mapping generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_r_stub_mapping(syntax_file: Path, mapping_file: Path) -> None:
    """Create a mapping CSV for the R-syntax-file conversion path.

    Parses the R syntax file to extract all ``names(data)[N] <- "code"``
    assignments, then auto-detects known column patterns using
    ``_AUTO_DETECT_R_MAP``.  Unknown columns are marked ``skip``; the user
    can change them as needed.
    """
    import re as _re

    # Parse all column codes in order from the syntax file
    col_codes: list[str] = []
    with syntax_file.open(encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            m = _re.search(r'names\(data\)\[\d+\]\s*<-\s*"([^"]+)"', line)
            if m:
                col_codes.append(m.group(1))

    rows = []
    for code in col_codes:
        target, transform = _AUTO_DETECT_R_MAP.get(code, ("", "skip"))
        rows.append({
            "question_code": code,
            "csv_column":    code,   # R CSV header == code, reference only
            "target_row":    target,
            "transform":     transform,
        })

    out = pd.DataFrame(rows, columns=["question_code", "csv_column",
                                      "target_row", "transform"])

    # Synthetic fixed-value row (Signature)
    synthetic = pd.DataFrame([
        {"question_code": "_signature", "csv_column": "",
         "target_row": "Signature", "transform": "fixed:1"},
    ], columns=["question_code", "csv_column", "target_row", "transform"])
    out = pd.concat([out, synthetic], ignore_index=True)

    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with mapping_file.open("w") as fh:
        fh.write("# prefix: XX26   ← EDIT THIS (e.g. DE26, IR25)\n")
        fh.write("# NOTE: This mapping is for the R syntax file export path.\n")
        fh.write("#       question_code values match R column names (underscores, not brackets).\n")
        fh.write("#       Ratio is computed automatically (Years outside / Age).\n")
        fh.write("#       Lexical items use transform=lexical_value (LIA1..LIA6 → -2..2/NX).\n")
        out.to_csv(fh, sep=";", index=False)
    print(f"  📄 Stub R mapping file created: {mapping_file}")
    print(f"     ({len(col_codes)} columns auto-detected from syntax file)")


def _load_mapping(mapping_file: Path) -> pd.DataFrame:
    """Load the survey-specific mapping CSV."""
    return pd.read_csv(str(mapping_file), sep=";", dtype=str,
                       comment="#").fillna("")


# ═══════════════════════════════════════════════════════════════════════════
#  Conversion
# ═══════════════════════════════════════════════════════════════════════════

def _apply_row(row: pd.Series, entries: list[dict], code_to_col: dict,
               target_row: str = "") -> str:
    """Compute the output value for one template row from its mapping entries.

    ``entries`` is a list of mapping rows all sharing the same target_row.
    """
    transform_types = {e["transform"] for e in entries}

    # ── Fixed value (no CSV column needed) ────────────────────────────────
    if any(t.startswith("fixed:") for t in transform_types):
        for e in entries:
            if e["transform"].startswith("fixed:"):
                return e["transform"][len("fixed:"):]
        return ""

    # ── Lexical value ─────────────────────────────────────────────────────
    if "lexical_value" in transform_types:
        for e in entries:
            if e["transform"] == "lexical_value" and e["question_code"] in code_to_col:
                raw = _clean_val(row.get(code_to_col[e["question_code"]]))
                if not raw:
                    return ""  # empty → NA
                mapped = _LEXICAL_VALUE_MAP.get(raw, raw)
                if mapped == "NX":
                    return "NX"
                # Reversed items: target row label ends with " -"
                if target_row.rstrip().endswith("-"):
                    try:
                        return str(int(mapped) * -1)
                    except (ValueError, TypeError):
                        pass
                return mapped
        return ""

    # ── Grammar value (raw read; numeric mapping applied by caller) ──────
    if "grammar_value" in transform_types:
        for e in entries:
            if e["transform"] == "grammar_value" and e["question_code"] in code_to_col:
                return _clean_val(row.get(code_to_col[e["question_code"]]))
        return ""

    # ── Language checkboxes ───────────────────────────────────────────────
    if any(t.startswith("lang_checkbox:") for t in transform_types):
        langs = []
        for e in entries:
            t = e["transform"]
            code = e["question_code"]
            if t.startswith("lang_checkbox:") and code in code_to_col:
                label = t[len("lang_checkbox:"):]
                if _clean_val(row.get(code_to_col[code])).lower() == "yes":
                    langs.append(label)
            elif t == "lang_other" and code in code_to_col:
                v = _clean_val(row.get(code_to_col[code]))
                if v:
                    langs.append(v)
        return "; ".join(langs)

    # ── Qualification checkboxes ──────────────────────────────────────────
    if any(t.startswith("qual_checkbox:") for t in transform_types):
        quals = []
        other_parts = []
        for e in entries:
            t = e["transform"]
            code = e["question_code"]
            if t.startswith("qual_checkbox:") and code in code_to_col:
                label = t[len("qual_checkbox:"):]
                if _clean_val(row.get(code_to_col[code])).lower() in ("yes", "y"):
                    quals.append(label)
            elif t == "qual_other" and code in code_to_col:
                v = _clean_val(row.get(code_to_col[code]))
                if v:
                    other_parts.append(v)
        all_parts = quals + other_parts
        return "; ".join(all_parts)

    # ── Base + other (combine_other pattern) ──────────────────────────────
    base_entries = [e for e in entries if e["transform"] == ""]
    other_entries = [e for e in entries if e["transform"] == "other"]
    if base_entries or other_entries:
        base = ""
        for e in base_entries:
            if e["question_code"] in code_to_col:
                base = _clean_val(row.get(code_to_col[e["question_code"]]))
                break
        other = ""
        for e in other_entries:
            if e["question_code"] in code_to_col:
                other = _clean_val(row.get(code_to_col[e["question_code"]]))
                break
        if other:
            return f"{base}: {other}".strip(": ") if base else other
        return base

    # ── qual_other only (lexical comments collected) ──────────────────────
    if transform_types == {"qual_other"}:
        parts = []
        for e in entries:
            if e["question_code"] in code_to_col:
                v = _clean_val(row.get(code_to_col[e["question_code"]]))
                if v:
                    parts.append(v)
        return " | ".join(parts)

    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  R syntax file conversion path
# ═══════════════════════════════════════════════════════════════════════════

def convert_limesurvey_r(
    data_csv: str | Path,
    syntax_file: str | Path,
    informant_prefix: str,
) -> Path | None:
    """Convert a LimeSurvey R-export pair to the transposed XLSX format.

    Workflow
    --------
    1. The R syntax file is executed via ``Rscript`` (using
       ``lib/r_scripts/run_limesurvey_syntax.R``).  This produces an
       intermediate CSV with R-style column names (e.g. ``PI01``,
       ``SSA_SSA1``, ``LI01_SQ001``) and human-readable labels for grammar
       items.  Lexical columns are restored to their raw ``LIA1``-``LIA6``
       codes so the direction-preserving ``lexical_value`` transform works
       correctly.
    2. The intermediate CSV is consumed by the same mapping/transform logic
       used by :func:`convert_limesurvey`.  The mapping file uses R column
       codes in the ``question_code`` column.

    Parameters
    ----------
    data_csv :
        Path to the LimeSurvey R data CSV
        (e.g. ``survey_998353_R_data_file.csv``).
    syntax_file :
        Path to the matching R syntax file
        (e.g. ``survey_998353_R_syntax_file.R``).
    informant_prefix :
        Prefix for InformantIDs, e.g. ``'DE26'``.

    Returns
    -------
    Path to the generated XLSX, or ``None`` if conversion was skipped
    (e.g. because a mapping stub was just generated).
    """
    import subprocess
    import tempfile

    data_csv    = Path(data_csv)
    syntax_file = Path(syntax_file)
    survey_r_id = _survey_r_id_from_path(data_csv)
    mapping_file = _mapping_path(survey_r_id)

    print(f"\n  Data CSV:    {data_csv.name}")
    print(f"  Syntax file: {syntax_file.name}")
    print(f"  Survey ID:   {survey_r_id}")
    print(f"  Prefix:      {informant_prefix}")

    # ── Guard: mapping file must exist ────────────────────────────────────
    if not mapping_file.exists():
        _generate_r_stub_mapping(syntax_file, mapping_file)
        print(f"  ⚠  No R mapping file found for {survey_r_id}.")
        print(f"     A stub has been created at:")
        print(f"       {mapping_file}")
        print(f"     Review and adjust target_row / transform columns,")
        print(f"     then re-run the conversion.")
        return None

    print(f"  Mapping:     {mapping_file.name}")

    # ── Step 1: Run R syntax file to produce intermediate CSV ─────────────
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        intermediate_csv = Path(tmp.name)

    print(f"  Running R syntax file …")
    result = subprocess.run(
        ["Rscript", "--no-save", "--no-restore",
         str(_R_SYNTAX_RUNNER),
         str(data_csv.resolve()),
         str(syntax_file.resolve()),
         str(intermediate_csv)],
        capture_output=True,
        text=True,
    )
    # Echo R output
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            print(f"    {line}")

    if result.returncode != 0:
        print(f"  ❌ R script failed (exit code {result.returncode})")
        intermediate_csv.unlink(missing_ok=True)
        return None

    # ── Step 2: Read intermediate CSV ─────────────────────────────────────
    df = pd.read_csv(str(intermediate_csv), dtype=str)
    intermediate_csv.unlink(missing_ok=True)
    n_respondents = len(df)
    print(f"  Respondents: {n_respondents}")

    # For the intermediate CSV, column name IS the question code directly.
    # Build code→col map (trivial identity mapping, but keeps _apply_row reusable).
    code_to_col: dict[str, str] = {col: col for col in df.columns}

    # ── Step 3: Load mapping ──────────────────────────────────────────────
    mapping_df = _load_mapping(mapping_file)

    active = mapping_df[
        (mapping_df["transform"] != "skip") &
        (mapping_df["target_row"].str.strip() != "")
    ].copy()

    from collections import defaultdict
    target_entries: dict[str, list[dict]] = defaultdict(list)
    for _, mrow in active.iterrows():
        target_entries[mrow["target_row"]].append({
            "question_code": mrow["question_code"],
            "transform":     mrow["transform"],
        })

    # ── Step 4: Load template row order ───────────────────────────────────
    template_rows = _load_template_rows()
    data_rows = template_rows[1:]

    _SEPARATORS = {"", "Lexical sets", "Comments",
                   "Grammar section 1", "Grammar section 2"}

    # ── Step 5: Process each respondent ───────────────────────────────────
    result_data: dict[str, dict[str, str]] = {}

    for idx, (_, row) in enumerate(df.iterrows()):
        informant_id = f"{informant_prefix}-{idx + 1:04d}"
        record: dict[str, str] = {}

        for target_row in data_rows:
            if target_row in _SEPARATORS:
                continue

            if target_row in target_entries:
                entries = target_entries[target_row]
                val = _apply_row(row, entries, code_to_col, target_row)

                if any(e["transform"] == "grammar_value" for e in entries):
                    val = _GRAMMAR_VALUE_MAP.get(val, val)

                # Empty grammar or lexical values → NA
                if not val and any(
                    e["transform"] in ("grammar_value", "lexical_value")
                    for e in entries
                ):
                    val = "NA"

                record[target_row] = val
            else:
                record[target_row] = "NA" if target_row == "Additional Varieties" else "ND"

        # Compute Ratio (Years outside / Age)
        years_outside = record.get("Years lived outside of home country", "")
        age = record.get("Age", "")
        if not years_outside:
            record["Ratio"] = "1"
        elif not age:
            record["Ratio"] = "NA"
        else:
            try:
                ratio = 1 - float(years_outside) / float(age)
                record["Ratio"] = str(round(ratio, 2))
            except (ValueError, ZeroDivisionError):
                record["Ratio"] = "NA"

        # Compute Years lived in home country (Age − Years outside)
        if not age:
            record["Years lived in home country"] = "NA"
        else:
            try:
                age_f = float(age)
                outside_f = float(years_outside) if years_outside else 0.0
                record["Years lived in home country"] = str(round(age_f - outside_f, 4))
            except (ValueError, ZeroDivisionError):
                record["Years lived in home country"] = "NA"

        # Compute Years lived in Main Variety (Age − Years outside)
        # and Ratio Main Variety (Years in Main Variety / Age)
        if not age:
            record["Years lived in Main Variety"] = "NA"
            record["Ratio Main Variety"] = "NA"
        else:
            try:
                age_f = float(age)
                outside_f = float(years_outside) if years_outside else 0.0
                years_main = age_f - outside_f
                record["Years lived in Main Variety"] = str(round(years_main, 4))
                record["Ratio Main Variety"] = str(round(years_main / age_f, 4))
            except (ValueError, ZeroDivisionError):
                record["Years lived in Main Variety"] = "NA"
                record["Ratio Main Variety"] = "NA"

        for k in record:
            if record[k].lower() == "nan":
                record[k] = ""

        result_data[informant_id] = record

    # ── Step 6: Build transposed DataFrame ───────────────────────────────
    informant_ids = list(result_data.keys())
    output_rows = []
    for row_label in template_rows:
        row_vals = [row_label if row_label != "" else ""]
        if row_label in _SEPARATORS or row_label == "Informant ID":
            row_vals.extend([""] * n_respondents)
        else:
            for iid in informant_ids:
                row_vals.append(result_data[iid].get(row_label, "ND"))
        output_rows.append(row_vals)

    columns = ["Informant ID"] + informant_ids
    out_df = pd.DataFrame(output_rows, columns=columns)
    out_df = out_df.fillna("").replace("nan", "")

    # ── Step 7: Write XLSX ────────────────────────────────────────────────
    out_name = f"{informant_prefix}_all_data_final.xlsx"
    CONVERSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONVERSION_OUTPUT_DIR / out_name
    out_df.to_excel(str(out_path), index=False, engine="openpyxl")

    n_mapped = len([r for r in data_rows if r not in _SEPARATORS
                    and r in target_entries])
    n_nd = len([r for r in data_rows if r not in _SEPARATORS
                and r not in target_entries])
    print(f"  ✅ Written to {out_path.name}")
    print(f"     Rows: {len(template_rows)-1} template rows  "
          f"({n_mapped} mapped, {n_nd} filled with ND)")

    # ── Step 8: Archive input files ───────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    done_dir = CONVERSION_INPUT_DIR / "Done" / timestamp
    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(data_csv),    str(done_dir / data_csv.name))
    shutil.move(str(syntax_file), str(done_dir / syntax_file.name))
    print(f"  📁 Input files archived to {done_dir.relative_to(DATA_DIR.parent)}/")

    return out_path


def convert_all_limesurvey_exports() -> list[Path]:
    """Find and convert all LimeSurvey R-export pairs in data/conversion/input/.

    Each pair consists of ``survey_NNN_R_data_file.csv`` and the matching
    ``survey_NNN_R_syntax_file.R`` in the same directory.

    The informant prefix is read from the mapping file's ``# prefix:`` comment.

    Returns a list of successfully generated XLSX paths.
    """
    print()
    print("=" * 80)
    print("  STAGE: Convert LimeSurvey exports → XLSX")
    print("=" * 80)

    if not CONVERSION_INPUT_DIR.exists():
        print(f"  ⚠  Input directory not found: {CONVERSION_INPUT_DIR}")
        return []

    if not TEMPLATE_XLSX.exists():
        print(f"  ❌ Template XLSX not found: {TEMPLATE_XLSX}")
        return []

    r_data_files = sorted(
        f for f in CONVERSION_INPUT_DIR.glob("*.csv")
        if re.search(r'survey_\d+_R_data_file', f.name, re.IGNORECASE)
    )

    if not r_data_files:
        print(f"  ⚠  No R data files (survey_*_R_data_file.csv) found in {CONVERSION_INPUT_DIR}")
        return []

    print(f"  Found {len(r_data_files)} R data file(s)")

    generated: list[Path] = []
    skipped:   list[str]  = []

    for data_csv in r_data_files:
        syntax_file = _r_syntax_file_for(data_csv)
        if syntax_file is None:
            print(f"  ⚠  No matching R syntax file for {data_csv.name} – skipping")
            skipped.append(data_csv.name)
            continue

        survey_r_id  = _survey_r_id_from_path(data_csv)
        mapping_file = _mapping_path(survey_r_id)
        prefix       = _read_prefix_from_mapping(mapping_file)

        out_path = convert_limesurvey_r(data_csv, syntax_file, prefix)
        if out_path is not None:
            generated.append(out_path)
        else:
            skipped.append(data_csv.name)

    print()
    if generated:
        print(f"  ✅ Converted: {len(generated)} file(s)")
    if skipped:
        print(f"  ⚠  Skipped (stub generated / no syntax file): {len(skipped)} file(s)")
        for name in skipped:
            print(f"       {name}")
    return generated


def _read_prefix_from_mapping(mapping_file: Path) -> str:
    """Read the informant prefix from a ``# prefix: XX26`` comment line."""
    if not mapping_file.exists():
        return "XX26"
    with mapping_file.open() as fh:
        for line in fh:
            m = re.match(r"#\s*prefix\s*:\s*(\S+)", line.strip())
            if m:
                return m.group(1)
    return "XX26"

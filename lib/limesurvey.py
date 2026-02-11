"""
LimeSurvey Conversion Module
=============================
Converts LimeSurvey CSV exports into the transposed XLSX format
expected by the existing ETL pipeline.

Handles three survey variants:
  - Ireland (lexical only): PI06 checkboxes for Irish/English
  - Germany lexical: PI06 dropdown, EP06[SQ001-6] combined parent/partner
  - Germany full (grammar + lexical): PI06 checkboxes for many languages,
    PI07m/PI07f, EP06/EP07/EP08/EP09, grammar spoken SSA-SSF, grammar
    written WSG-WSN

The column mapping is loaded from data/mappings/limesurvey_column_mapping.csv.
"""

import re
import pandas as pd
from pathlib import Path

from lib import DATA_DIR, INPUT_DIR, MAPPINGS_DIR


LIMESURVEY_DIR = INPUT_DIR / "Limesurvey_export"
MAPPING_CSV = MAPPINGS_DIR / "limesurvey_column_mapping.csv"

# EP04 qualification labels by subquestion code
_EP04_LABELS = {
    "SQ001": "Apprenticeship",
    "SQ002": "Vocational classes",
    "SQ007": "General secondary education",
    "SQ003": "Bachelor",
    "SQ004": "Master's",
    "SQ005": "PhD",
    "SQ006": "Other",
}

# PI06 language checkbox labels per survey variant
_PI06_LANG_LABELS_IRELAND = {
    "SQ006": "English",
    "SQ001": "Irish",
}

_PI06_LANG_LABELS_DE_FULL = {
    "SQ001SQ001": "German",
    "SQ002": "English",
    "SQ003": "French",
    "SQ004": "Italian",
    "SQ005": "Spanish",
    "SQ006": "Hungarian",
    "SQ007": "Polish",
    "SQ008": "Czech",
}

# PI07m / PI07f checkbox labels (Germany full survey)
_PI07_LANG_LABELS = {
    "SQ001SQ001": "German",
    "SQ002": "English",
    "SQ003": "French",
    "SQ004": "Italian",
    "SQ005": "Spanish",
    "SQ006": "Hungarian",
    "SQ007": "Polish",
    "SQ008": "Czech",
}

# Grammar answer text → numeric scale
_GRAMMAR_VALUE_MAP = {
    "No-one": "0",
    "Few": "1",
    "Some": "2",
    "Many": "3",
    "Most": "4",
    "Everyone": "5",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_question_code(header: str) -> str:
    """Extract the question code from a LimeSurvey column header.

    Headers are formatted as:
      'QuestionCode. Questiontext'
      'QuestionCode[Subquestion]. Questiontext'

    Returns the part before the first '. ' (dot-space).
    Strips any leading/trailing quotes.
    """
    header = header.strip().strip('"')
    match = re.match(r'^([^.]+)\.\s', header)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole header stripped
    return header.strip()


def _load_mapping() -> dict[str, str]:
    """Load the question_code → target_column mapping from CSV.

    Returns a dict: { question_code: target_column }.
    """
    mapping = {}
    df = pd.read_csv(str(MAPPING_CSV), sep=";", dtype=str)
    for _, row in df.iterrows():
        mapping[row["question_code"]] = row["target_column"]
    return mapping


def _combine_other(base_val, other_val) -> str:
    """Combine a base value with an 'other' value if non-empty."""
    base = str(base_val).strip() if pd.notna(base_val) else ""
    other = str(other_val).strip() if pd.notna(other_val) else ""
    if other:
        return f"{base}: {other}".strip(": ") if base else other
    return base


def _combine_language_checkboxes(row: pd.Series, code_to_col: dict,
                                 base_code: str = "PI06",
                                 lang_labels: dict | None = None) -> str:
    """Combine PI06 (or similar) checkbox columns into a languages string.

    Parameters
    ----------
    base_code : The question base code (e.g. 'PI06', 'PI07m', 'PI07f').
    lang_labels : dict mapping subquestion codes to language names.
    """
    if lang_labels is None:
        lang_labels = _PI06_LANG_LABELS_IRELAND

    langs = []
    for sq_code, lang_label in lang_labels.items():
        full_code = f"{base_code}[{sq_code}]"
        if full_code in code_to_col:
            val = row.get(code_to_col[full_code], "")
            if str(val).strip().lower() == "yes":
                langs.append(lang_label)

    other_code = f"{base_code}[other]"
    if other_code in code_to_col:
        other_val = str(row.get(code_to_col[other_code], "")).strip()
        if other_val and other_val.lower() != "nan":
            langs.append(other_val)

    return "; ".join(langs) if langs else ""


def _combine_language_dropdown(row: pd.Series, code_to_col: dict) -> str:
    """Combine Germany PI06 dropdown + comment into a language string."""
    base_code = "PI06"
    comment_code = "PI06[comment]"

    base = ""
    if base_code in code_to_col:
        base = str(row.get(code_to_col[base_code], "")).strip()
        if base.lower() == "nan":
            base = ""

    comment = ""
    if comment_code in code_to_col:
        comment = str(row.get(code_to_col[comment_code], "")).strip()
        if comment.lower() == "nan":
            comment = ""

    if comment:
        return f"{base}; {comment}".strip("; ") if base else comment
    return base


def _combine_qualifications(row: pd.Series, code_to_col: dict) -> str:
    """Combine Germany EP04 checkbox qualifications into a single string.

    Looks for EP04[SQ001]-EP04[SQ006] (plus SQ007). Each has a yes/no value
    and an optional comment suffix. Collects those marked 'Yes'.
    """
    quals = []
    for sq_code, label in _EP04_LABELS.items():
        full_code = f"EP04[{sq_code}]"
        comment_code = f"EP04[{sq_code}comment]"

        if full_code in code_to_col:
            val = str(row.get(code_to_col[full_code], "")).strip()
            if val.lower() in ("yes", "y"):
                # Check for a comment
                comment = ""
                if comment_code in code_to_col:
                    comment = str(row.get(code_to_col[comment_code], "")).strip()
                    if comment.lower() == "nan":
                        comment = ""
                if comment:
                    quals.append(f"{label} ({comment})")
                else:
                    quals.append(label)

    return "; ".join(quals) if quals else ""


def _combine_lexical_comments(row: pd.Series, code_to_col: dict) -> str:
    """Combine LI1C, L2C, L3C, L4C into a single CommentsLexical string."""
    parts = []
    for code in ("LI1C", "L2C", "L3C", "L4C"):
        if code in code_to_col:
            val = str(row.get(code_to_col[code], "")).strip()
            if val and val.lower() != "nan":
                parts.append(val)
    return " | ".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════════
#  Main conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_limesurvey(csv_path: str | Path, informant_prefix: str,
                       survey_type: str = "lexical") -> Path:
    """Convert a LimeSurvey CSV export to the transposed XLSX format.

    Parameters
    ----------
    csv_path : path to the LimeSurvey CSV export file.
    informant_prefix : prefix for InformantIDs, e.g. 'IR25' or 'DE24'.
        Will be used to generate IDs like 'IR25-0001', 'IR25-0002', …
    survey_type : 'lexical' or 'grammar'. Affects the output filename.

    Returns
    -------
    Path to the generated XLSX file in data/input/.
    """
    csv_path = Path(csv_path)
    print(f"\n  Converting LimeSurvey export: {csv_path.name}")
    print(f"  Informant prefix: {informant_prefix}")

    # ── Read CSV and build code→column-index mapping ──────────────────────
    df = pd.read_csv(str(csv_path), dtype=str)

    # Build a dict: question_code → original CSV column name
    code_to_col: dict[str, str] = {}
    for col in df.columns:
        code = _extract_question_code(col)
        code_to_col[code] = col

    # ── Load mapping ──────────────────────────────────────────────────────
    mapping = _load_mapping()

    # ── Detect survey variant ─────────────────────────────────────────────
    has_pi06_sq001sq001 = "PI06[SQ001SQ001]" in code_to_col  # Germany full
    has_pi06_checkboxes = ("PI06[SQ006]" in code_to_col
                           or "PI06[SQ001]" in code_to_col)
    has_pi06_dropdown = ("PI06" in code_to_col
                         and not has_pi06_checkboxes
                         and not has_pi06_sq001sq001)
    has_ep04 = "EP04[SQ001]" in code_to_col  # Germany qualifications
    has_pi08copy = "PI08Copy" in code_to_col  # Ireland years-in-country
    has_grammar = "SSA[SSA1]" in code_to_col  # Grammar sections present
    has_pi07m = "PI07m[SQ002]" in code_to_col  # Germany full mother lang
    has_ep09 = "EP09[SQ002]" in code_to_col  # Germany full parent occup

    if has_pi06_sq001sq001:
        variant = "Germany_full"
    elif has_pi06_dropdown:
        variant = "Germany"
    else:
        variant = "Ireland"
    print(f"  Detected survey variant: {variant}")
    if has_grammar:
        print(f"  Grammar sections detected (spoken + written)")

    # ── Define the xlsx row labels in order ───────────────────────────────
    # The order follows the existing xlsx files exactly
    sociodem_rows = [
        "Age",
        "Gender",
        "Nationality",
        "Ethnic self-identification",
        "Country or region you identify with",
        "Languages used at home while  growing up",
        "Native Lg. Mother",
        "Native Lg. Father",
        "Primary school",
        "Secondary school",
        "Name and Place of high school",
        "Qualifications",
        "Current occupation",
        "Qualification mother",
        "Occupation mother",
        "Qualification father",
        "Occupation father",
        "Qualification partner",
        "Occupation partner",
        "Years lived outside of home country",
        "Timeline Comments",
        "Years lived in home country",
        "Ratio",
        "Years lived in other English-speaking countries",
        "Comments",
        "Signature",
        "Main Variety",
        "Additional Varieties",
        "Years lived in Main Variety",
        "Ratio Main Variety",
    ]

    lexical_rows = [
        "a drop in the ocean -",
        "a tap",
        "aluminium",
        "anticlockwise -",
        "aubergine",
        "autumn",
        "backwards",
        "bicentenary -",
        "biscuit",
        "bookings -",
        "boot",
        "car park -",
        "centre",
        "chemist's -",
        "ill -",
        "potato chips",
        "chips",
        "cinema -",
        "colour",
        "cupboard -",
        "driving licence",
        "dummy -",
        "dustbin",
        "fish fingers -",
        "football",
        "forwards -",
        "globalisation",
        "glocalisation -",
        "holiday",
        "liberalisation",
        "jacket potato",
        "laund(e)rette -",
        "potato crisps -",
        "crisps -",
        "to licence -",
        "lift",
        "localisation -",
        "lorry",
        "maths -",
        "mobile phone",
        "modernisation -",
        "nappies",
        "organisation -",
        "parcel",
        "pavement -",
        "petrol",
        "petrol station -",
        "postman",
        "pushchair -",
        "railway",
        "realisation -",
        "roundabout",
        "rubber -",
        "rubbish",
        "shopping trolley -",
        "sport",
        "storm in a teacup -",
        "subway",
        "to let -",
        "torch",
        "touch wood -",
        "trainers",
        "whilst -",
        "windscreen",
        "a book about  chemistry -",
        "compare X to Y -",
        "typical of -",
        "Anyway",
    ]

    all_rows = sociodem_rows + ["", "Lexical sets"] + lexical_rows + ["", "Comments"]

    # ── Grammar row lists (only used when grammar sections present) ───────
    grammar_spoken_rows = []
    grammar_written_rows = []
    if has_grammar:
        # Spoken: A1-A23, B1-B23, C1-C23, D1-D23, E1-E23, F1-F23
        for letter in "ABCDEF":
            for n in range(1, 24):
                grammar_spoken_rows.append(f"{letter}{n}")

        # Written: G1-G26, H1-H26, I1-I26, J1-J26,
        #          K1-K3,K4a,K4b,K5-K26,
        #          L1-L26, M1-M25, N1-N25
        for letter in "GHIJ":
            for n in range(1, 27):
                grammar_written_rows.append(f"{letter}{n}")
        # K has K4a and K4b instead of K4
        for n in range(1, 4):
            grammar_written_rows.append(f"K{n}")
        grammar_written_rows.append("K4a")
        grammar_written_rows.append("K4b")
        for n in range(5, 27):
            grammar_written_rows.append(f"K{n}")
        # L: 1-26
        for n in range(1, 27):
            grammar_written_rows.append(f"L{n}")
        # M: 1-25
        for n in range(1, 26):
            grammar_written_rows.append(f"M{n}")
        # N: 1-25
        for n in range(1, 26):
            grammar_written_rows.append(f"N{n}")

        all_rows = (sociodem_rows
                    + ["", "Lexical sets"] + lexical_rows
                    + ["", "Comments"]
                    + ["", "Grammar section 1"]
                    + grammar_spoken_rows
                    + ["", "Grammar section 2"]
                    + grammar_written_rows)

    # Build the reverse mapping: target_column → question_code
    target_to_code: dict[str, str] = {}
    for qcode, target in mapping.items():
        if not target.startswith("_"):
            target_to_code[target] = qcode

    # ── Process each respondent ───────────────────────────────────────────
    n_respondents = len(df)
    print(f"  Processing {n_respondents} respondents")

    result_data = {}  # { informant_id: { row_label: value } }

    for idx, (_, row) in enumerate(df.iterrows()):
        informant_id = f"{informant_prefix}-{idx + 1:04d}"
        record: dict[str, str] = {}

        # ── Simple 1:1 mapped columns ────────────────────────────────────
        for target, qcode in target_to_code.items():
            if qcode in code_to_col:
                val = str(row.get(code_to_col[qcode], "")).strip()
                if val.lower() == "nan":
                    val = ""
                record[target] = val

        # ── Gender: combine with other ───────────────────────────────────
        record["Gender"] = _combine_other(
            row.get(code_to_col.get("PI02", ""), ""),
            row.get(code_to_col.get("PI02[other]", ""), ""),
        )

        # ── Nationality: combine with other ──────────────────────────────
        nat_val = row.get(code_to_col.get("PI03", ""), "")
        nat_other = row.get(code_to_col.get("PI03[other]", ""), "")
        record["Nationality"] = _combine_other(nat_val, nat_other)

        # ── Languages at home ────────────────────────────────────────────
        if has_pi06_sq001sq001:
            record["Languages used at home while  growing up"] = (
                _combine_language_checkboxes(row, code_to_col,
                                            "PI06", _PI06_LANG_LABELS_DE_FULL)
            )
        elif has_pi06_checkboxes:
            record["Languages used at home while  growing up"] = (
                _combine_language_checkboxes(row, code_to_col,
                                            "PI06", _PI06_LANG_LABELS_IRELAND)
            )
        elif has_pi06_dropdown:
            record["Languages used at home while  growing up"] = (
                _combine_language_dropdown(row, code_to_col)
            )

        # ── Mother's / Father's native language ─────────────────────────
        if has_pi07m:
            record["Native Lg. Mother"] = _combine_language_checkboxes(
                row, code_to_col, "PI07m", _PI07_LANG_LABELS)
            record["Native Lg. Father"] = _combine_language_checkboxes(
                row, code_to_col, "PI07f", _PI07_LANG_LABELS)

        # ── Primary school: combine with other ───────────────────────────
        record["Primary school"] = _combine_other(
            row.get(code_to_col.get("EP01", ""), ""),
            row.get(code_to_col.get("EP01[other]", ""), ""),
        )

        # ── Secondary school (Germany only) ──────────────────────────────
        if "EP02" in code_to_col:
            record["Secondary school"] = _combine_other(
                row.get(code_to_col.get("EP02", ""), ""),
                row.get(code_to_col.get("EP02[other]", ""), ""),
            )

        # ── Qualifications ───────────────────────────────────────────────
        if has_ep04:
            record["Qualifications"] = _combine_qualifications(row, code_to_col)

        # ── Occupation: combine with other ───────────────────────────────
        record["Current occupation"] = _combine_other(
            row.get(code_to_col.get("EP05", ""), ""),
            row.get(code_to_col.get("EP05[other]", ""), ""),
        )

        # ── Timeline Comments from PI10 ──────────────────────────────────
        if "PI10" in code_to_col:
            val = str(row.get(code_to_col["PI10"], "")).strip()
            record["Timeline Comments"] = "" if val.lower() == "nan" else val

        # ── Years lived in home country (Ireland: PI08Copy) ──────────────
        if has_pi08copy and "PI08Copy" in code_to_col:
            val = str(row.get(code_to_col["PI08Copy"], "")).strip()
            record["Years lived in home country"] = "" if val.lower() == "nan" else val

        # ── Germany full: EP06/EP07/EP08 with [other] ────────────────────
        if has_ep09:
            # Qualification mother (EP06 dropdown + other)
            if "EP06" in code_to_col and "EP06[SQ001]" not in code_to_col:
                record["Qualification mother"] = _combine_other(
                    row.get(code_to_col.get("EP06", ""), ""),
                    row.get(code_to_col.get("EP06[other]", ""), ""),
                )
            # Qualification father (EP07 dropdown + other)
            if "EP07" in code_to_col:
                record["Qualification father"] = _combine_other(
                    row.get(code_to_col.get("EP07", ""), ""),
                    row.get(code_to_col.get("EP07[other]", ""), ""),
                )
            # Qualification partner (EP08 dropdown + other)
            if "EP08" in code_to_col:
                record["Qualification partner"] = _combine_other(
                    row.get(code_to_col.get("EP08", ""), ""),
                    row.get(code_to_col.get("EP08[other]", ""), ""),
                )
            # Occupations from EP09 subquestions
            for sq, target in [("SQ002", "Occupation mother"),
                                ("SQ004", "Occupation father"),
                                ("SQ006", "Occupation partner")]:
                ep09_code = f"EP09[{sq}]"
                if ep09_code in code_to_col:
                    val = str(row.get(code_to_col[ep09_code], "")).strip()
                    record[target] = "" if val.lower() == "nan" else val

        # ── Years in English-speaking countries (PI11) ───────────────────
        if "PI11" in code_to_col:
            val = str(row.get(code_to_col["PI11"], "")).strip()
            record["Years lived in other English-speaking countries"] = (
                "" if val.lower() == "nan" else val
            )

        # ── CommentsLexical ──────────────────────────────────────────────
        comments = _combine_lexical_comments(row, code_to_col)
        record["Comments"] = comments

        # ── Grammar values: convert text labels to numeric ───────────────
        if has_grammar:
            # GrammarSpokenFillingInFor from HC2
            if "HC2" in code_to_col:
                val = str(row.get(code_to_col["HC2"], "")).strip()
                record["GrammarSpokenFillingInFor"] = (
                    "" if val.lower() == "nan" else val
                )

            for label in grammar_spoken_rows + grammar_written_rows:
                if label in record:
                    raw = record[label]
                    record[label] = _GRAMMAR_VALUE_MAP.get(raw, raw)

        # ── Empty columns that don't come from the survey ────────────────
        for col in ("Signature", "Main Variety", "Additional Varieties",
                     "Years lived in Main Variety", "Ratio Main Variety",
                     "Ratio", "Years lived in other English-speaking countries"):
            record.setdefault(col, "")

        # Ensure no 'nan' strings remain
        for key in record:
            if str(record[key]).lower() == "nan":
                record[key] = ""

        result_data[informant_id] = record

    # ── Build the transposed output DataFrame ─────────────────────────────
    # Column 0 = "Informant ID" (row labels), then one column per respondent
    informant_ids = list(result_data.keys())

    output_rows = []
    for row_label in all_rows:
        row_values = [row_label]
        if row_label == "" or row_label in ("Lexical sets", "Comments",
                                             "Grammar section 1",
                                             "Grammar section 2"):
            # Separator/header rows: empty values
            row_values.extend([""] * n_respondents)
        else:
            for iid in informant_ids:
                row_values.append(result_data[iid].get(row_label, ""))
        output_rows.append(row_values)

    columns = ["Informant ID"] + informant_ids
    out_df = pd.DataFrame(output_rows, columns=columns)

    # Replace 'nan' strings and actual NaN with empty strings
    out_df = out_df.fillna("")
    out_df = out_df.replace("nan", "")

    # ── Write XLSX ────────────────────────────────────────────────────────
    out_name = f"{informant_prefix}_{survey_type}_online.xlsx"
    out_path = INPUT_DIR / out_name
    out_df.to_excel(str(out_path), index=False, engine="openpyxl")

    print(f"  ✅ Written {n_respondents} respondents to {out_path.name}")
    grammar_info = ""
    if has_grammar:
        grammar_info = (f" + {len(grammar_spoken_rows)} grammar spoken"
                        f" + {len(grammar_written_rows)} grammar written")
    print(f"     Rows: {len(all_rows)} ({len(sociodem_rows)} sociodem + "
          f"{len(lexical_rows)} lexical{grammar_info})")

    return out_path


def convert_all_limesurvey_exports() -> list[Path]:
    """Find and convert all LimeSurvey CSV exports in the export directory.

    Looks for Lexical*.csv and Grammar*.csv files.
    Detects the country from the filename and infers a prefix.
    Returns a list of generated XLSX paths.
    """
    print()
    print("=" * 80)
    print("  STAGE: Convert LimeSurvey exports → XLSX")
    print("=" * 80)

    if not LIMESURVEY_DIR.exists():
        print(f"  ⚠  No LimeSurvey export directory found at {LIMESURVEY_DIR}")
        return []

    csv_files = sorted(
        set(LIMESURVEY_DIR.glob("Lexical*.csv"))
        | set(LIMESURVEY_DIR.glob("Grammar*.csv"))
    )
    if not csv_files:
        print(f"  ⚠  No Lexical*.csv or Grammar*.csv files found in "
              f"{LIMESURVEY_DIR}")
        return []

    print(f"  Found {len(csv_files)} LimeSurvey export(s)")

    generated = []
    for csv_file in csv_files:
        # Infer prefix and survey type from filename
        prefix = _infer_prefix(csv_file)
        survey_type = "grammar" if csv_file.stem.lower().startswith("grammar") else "lexical"
        out_path = convert_limesurvey(csv_file, prefix, survey_type)
        generated.append(out_path)

    return generated


def _infer_prefix(csv_path: Path) -> str:
    """Infer a country+year InformantID prefix from the CSV filename.

    For 'Lexical_Germany_results-survey367676.csv' → 'DE25'
    For 'Lexical_results-survey367676.csv' → 'IR25'
    For 'Grammar_Germany_results.csv' → 'DE25'
    Other patterns: extract the country name and use current year.
    """
    stem = csv_path.stem.lower()

    # Known country keywords → code mapping
    country_codes = {
        "germany": "DE",
        "ireland": "IR",
        "australia": "AU",
        "canada": "CA",
        "denmark": "DK",
        "spain": "ES",
        "gibraltar": "GI",
        "hawaii": "HA",
        "india": "IN",
        "jersey": "JE",
        "malta": "MT",
        "newzealand": "NZ",
        "philippines": "PH",
        "puertorico": "PR",
        "scotland": "SC",
        "slovenia": "SL",
        "sweden": "SW",
        "uk": "UK",
        "us": "US",
    }

    detected_code = None
    for keyword, code in country_codes.items():
        if keyword in stem.replace("_", "").replace("-", ""):
            detected_code = code
            break

    if detected_code is None:
        # Try to extract from filename pattern Lexical_XYZ_results
        # or Grammar_XYZ_results
        match = re.search(r"(?:lexical|grammar)_(\w+?)_results", stem)
        if match:
            detected_code = match.group(1)[:2].upper()
        else:
            # Fallback for 'Lexical_results-...' (no country in name)
            detected_code = "IR"  # default to Ireland

    # Use current year (last two digits)
    import datetime
    year_short = datetime.datetime.now().strftime("%y")

    prefix = f"{detected_code}{year_short}"
    return prefix

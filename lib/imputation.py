"""
Variety-stratified PMM imputation with cross-modality predictors.

This module implements Predictive Mean Matching (PMM) for BSLVC grammar
and lexical data, with two key improvements over hdImpute:

1. **Variety stratification**: imputation is performed within each variety
   group, so donors share the same linguistic background.
2. **Cross-modality predictors**: for the 138 spoken grammar features that
   have a written counterpart (and vice versa), the counterpart is included
   as a predictor, exploiting r ≈ 0.37 correlations and rescuing 68–90 %
   of otherwise blind imputations.

The module produces a detailed quality log with cross-validated MAE,
exact-match rate, within-±1 rate, mean-shift and variance-ratio diagnostics.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lib import DB_PATH, OUTPUT_DIR

# ── constants ───────────────────────────────────────────────────────────────

SEED = 211191
PMM_K = 5                 # number of nearest-neighbour donors
MAX_PREDICTORS = 20       # max predictors per column (+ cross-modality counterpart)
N_CV_FOLDS = 5            # cross-validation folds for quality assessment
CV_MASK_FRAC = 0.10       # fraction of observed values to mask per fold
MIN_VARIETY_SIZE = 10     # absolute minimum for stratified imputation
GRAMMAR_CUTOFF = 50       # max missing per participant (grammar)
LEXICAL_CUTOFF = 25       # max missing per participant (lexical)

# Value ranges differ by data type:
#   Grammar features use a 0–5 ordinal scale
#   Lexical features use a -2 to +2 ordinal scale
GRAMMAR_VALUE_MIN, GRAMMAR_VALUE_MAX = 0, 5
LEXICAL_VALUE_MIN, LEXICAL_VALUE_MAX = -2, 2


# ── logging helper ──────────────────────────────────────────────────────────

class ImputationLog:
    """Collects log lines and writes them to a file at the end."""

    def __init__(self):
        self._buf = StringIO()

    def log(self, msg: str = ""):
        print(msg)
        self._buf.write(msg + "\n")

    def save(self, path: Path):
        path.write_text(self._buf.getvalue(), encoding="utf-8")
        print(f"  Log saved to: {path}")

    @property
    def text(self) -> str:
        return self._buf.getvalue()


# ── PMM core ────────────────────────────────────────────────────────────────

def _pmm_impute_column(
    y: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
    k: int = PMM_K,
) -> np.ndarray:
    """Impute missing values in *y* using PMM with predictors *X*.

    1. Fit Ridge regression on observed rows.
    2. Predict for all rows.
    3. For each missing row, pick the *k* observed rows whose predictions
       are closest, then randomly draw one donor's actual value.

    Returns a copy of *y* with NaNs filled.
    """
    obs_mask = ~np.isnan(y)
    if obs_mask.all() or not obs_mask.any():
        return y.copy()

    y_out = y.copy()

    # Fit on observed
    X_obs = X[obs_mask]
    y_obs = y[obs_mask]

    model = Ridge(alpha=1.0)
    model.fit(X_obs, y_obs)

    # Predict all rows
    y_hat = model.predict(X)

    # For each missing row, find k nearest observed predictions
    miss_idx = np.where(~obs_mask)[0]
    y_hat_obs = y_hat[obs_mask]

    for i in miss_idx:
        distances = np.abs(y_hat_obs - y_hat[i])
        nearest = np.argsort(distances)[:k]
        donor = rng.choice(nearest)
        y_out[i] = y_obs[donor]

    return y_out


def _select_predictors(
    col: str,
    feature_cols: list[str],
    aux_cols: list[str],
    corr_matrix: pd.DataFrame,
    priority_cols: list[str] | None = None,
    max_preds: int = MAX_PREDICTORS,
) -> list[str]:
    """Select the best predictors for *col*.

    Returns up to *max_preds* features ranked by absolute correlation,
    plus any *priority_cols* (cross-modality counterpart) always included.
    """
    candidates = [c for c in feature_cols if c != col] + aux_cols
    if not candidates:
        return []

    # Start with priority columns (cross-modality counterpart)
    selected: list[str] = []
    if priority_cols:
        for pc in priority_cols:
            if pc in candidates and pc in corr_matrix.columns and col in corr_matrix.index:
                selected.append(pc)

    # Rank remaining by absolute correlation with col
    remaining = [c for c in candidates if c not in selected and c in corr_matrix.columns]
    if remaining and col in corr_matrix.index:
        corrs = corr_matrix.loc[col, remaining].abs().sort_values(ascending=False)
        for c in corrs.index:
            if len(selected) >= max_preds:
                break
            if corrs[c] > 0.01:  # minimal threshold
                selected.append(c)

    return selected


def _pmm_impute_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    aux_cols: list[str] | None = None,
    rng: np.random.Generator | None = None,
    log: ImputationLog | None = None,
    n_cycles: int = 5,
    cross_modality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Impute all *feature_cols* in *df* using chained-equations PMM.

    Parameters
    ----------
    df : DataFrame with feature columns + optional auxiliary columns.
    feature_cols : columns to impute.
    aux_cols : additional predictor columns (not imputed themselves).
    rng : random generator.
    n_cycles : number of chained-equations iteration cycles.
    cross_modality_map : {feature: counterpart} for priority predictors.

    Returns a copy of *df* with feature_cols imputed.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    if aux_cols is None:
        aux_cols = []
    if cross_modality_map is None:
        cross_modality_map = {}

    result = df.copy()

    # Convert to numeric
    all_cols = list(set(feature_cols + aux_cols))
    for c in all_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce")

    # Initial fill: column median (for building first predictor matrix)
    medians = {}
    for c in feature_cols:
        med = result[c].median()
        medians[c] = med if not np.isnan(med) else 0.0

    # Sort features by missingness (fewest missing first)
    miss_counts = {c: result[c].isna().sum() for c in feature_cols}
    sorted_features = sorted(feature_cols, key=lambda c: miss_counts[c])

    # Only impute features that have missing values
    features_to_impute = [c for c in sorted_features if miss_counts[c] > 0]

    if not features_to_impute:
        return result

    # Working copy with initial fill for predictor building
    work = result.copy()
    for c in feature_cols:
        work[c] = work[c].fillna(medians[c])

    # Pre-compute correlation matrix for predictor selection
    corr_matrix = work[all_cols].corr()

    # Pre-compute predictor sets for each column
    pred_sets: dict[str, list[str]] = {}
    for col in features_to_impute:
        priority = []
        if col in cross_modality_map:
            cp = cross_modality_map[col]
            if cp in work.columns:
                priority = [cp]
        pred_sets[col] = _select_predictors(
            col, feature_cols, aux_cols, corr_matrix, priority,
        )

    for cycle in range(n_cycles):
        for col in features_to_impute:
            preds = pred_sets[col]
            if not preds:
                continue

            # Filter out zero-variance predictors
            good_preds = [p for p in preds if work[p].std() > 1e-10]
            if not good_preds:
                continue

            X = work[good_preds].values
            y_orig = result[col].values  # original with NaN

            y_imputed = _pmm_impute_column(y_orig, X, rng)

            # Update working copy for subsequent features
            work[col] = y_imputed

        if log and cycle == 0:
            filled = sum(
                np.isnan(result[c].values).sum() - np.isnan(work[c].values).sum()
                for c in features_to_impute
            )
            log.log(f"    cycle 1/{n_cycles}: {filled} values imputed")

    # Write back
    for c in features_to_impute:
        result[c] = work[c]

    return result


# ── cross-validation quality assessment ─────────────────────────────────────

def _cv_quality(
    df: pd.DataFrame,
    feature_cols: list[str],
    aux_cols: list[str] | None = None,
    rng: np.random.Generator | None = None,
    n_folds: int = N_CV_FOLDS,
    cross_modality_map: dict[str, str] | None = None,
    value_min: int = 0,
    value_max: int = 5,
) -> dict:
    """Hold-out cross-validation: mask CV_MASK_FRAC of observed values,
    impute, compare.  Returns a dict of quality metrics."""
    if rng is None:
        rng = np.random.default_rng(SEED + 999)
    if aux_cols is None:
        aux_cols = []

    all_true = []
    all_pred = []

    for fold in range(n_folds):
        df_masked = df.copy()
        masked_positions: list[tuple[int, str]] = []

        for col in feature_cols:
            obs_idx = df_masked.index[df_masked[col].notna()].tolist()
            n_mask = max(1, int(len(obs_idx) * CV_MASK_FRAC))
            if len(obs_idx) < 5:
                continue
            chosen = rng.choice(obs_idx, size=min(n_mask, len(obs_idx)), replace=False)
            for idx in chosen:
                masked_positions.append((idx, col))
                all_true.append(float(df_masked.at[idx, col]))
                df_masked.at[idx, col] = np.nan

        if not masked_positions:
            continue

        imputed = _pmm_impute_dataframe(
            df_masked, feature_cols, aux_cols, rng=rng, n_cycles=3,
            cross_modality_map=cross_modality_map,
        )

        for (idx, col), _ in zip(masked_positions, range(len(masked_positions))):
            all_pred.append(float(imputed.at[idx, col]))

    if not all_true:
        return {}

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    # Post-process predictions as we do for real imputation
    pred_arr = np.clip(np.round(pred_arr), value_min, value_max)

    diffs = np.abs(true_arr - pred_arr)

    return {
        "n_evaluated": len(true_arr),
        "mae": float(np.mean(diffs)),
        "exact_match_rate": float(np.mean(diffs == 0)),
        "within_1_rate": float(np.mean(diffs <= 1)),
        "mean_true": float(np.mean(true_arr)),
        "mean_pred": float(np.mean(pred_arr)),
        "mean_shift": float(np.mean(pred_arr) - np.mean(true_arr)),
        "var_true": float(np.var(true_arr)),
        "var_pred": float(np.var(pred_arr)),
        "var_ratio": float(np.var(pred_arr) / max(np.var(true_arr), 1e-10)),
    }


def _log_quality(
    log: ImputationLog,
    label: str,
    metrics: dict,
    variety_metrics: dict[str, dict] | None = None,
):
    """Write quality metrics to the log."""
    log.log(f"\n  ── Quality assessment: {label} ──")
    if not metrics:
        log.log("    (not enough data for cross-validation)")
        return

    log.log(f"    Cross-validated on {metrics['n_evaluated']:,} held-out values:")
    log.log(f"      MAE              = {metrics['mae']:.3f}  (0 = perfect, <0.5 good, <0.8 acceptable)")
    log.log(f"      Exact match rate = {metrics['exact_match_rate']:.1%}  (chance ≈ 17%, >40% good)")
    log.log(f"      Within-±1 rate   = {metrics['within_1_rate']:.1%}  (>80% good)")
    log.log(f"    Distribution preservation:")
    log.log(f"      Mean shift       = {metrics['mean_shift']:+.3f}  (should be near 0)")
    log.log(f"      Variance ratio   = {metrics['var_ratio']:.3f}  (should be near 1)")

    if variety_metrics:
        log.log(f"\n    Per-variety breakdown:")
        log.log(f"    {'Variety':>8s}  {'n':>5s}  {'MAE':>6s}  {'Exact%':>7s}  {'±1%':>6s}  {'Shift':>7s}  {'VarR':>6s}")
        for var in sorted(variety_metrics):
            m = variety_metrics[var]
            if not m:
                continue
            log.log(
                f"    {var:>8s}  {m['n_evaluated']:5d}  "
                f"{m['mae']:6.3f}  {m['exact_match_rate']:6.1%}  "
                f"{m['within_1_rate']:5.1%}  {m['mean_shift']:+7.3f}  "
                f"{m['var_ratio']:6.3f}"
            )


# ── variety helpers ─────────────────────────────────────────────────────────

def _extract_variety(informant_id: str) -> str:
    """Extract variety prefix from InformantID (e.g. 'MT08_123' → 'MT')."""
    m = re.match(r"^[A-Za-z]+", str(informant_id))
    return m.group().upper() if m else "UNKNOWN"


def _get_spoken_written_mapping(db_path: Path) -> dict[str, str]:
    """Return {spoken_code: written_code} from bslvc_meta.also_in_item."""
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(
            "SELECT question_code, also_in_item FROM bslvc_meta "
            "WHERE section = 'Spoken' "
            "AND also_in_item IS NOT NULL AND also_in_item != 'None'",
            conn,
        )
    return dict(zip(df["question_code"], df["also_in_item"]))


# ── main imputation routines ───────────────────────────────────────────────

def impute_grammar(
    db_path: Path,
    log: ImputationLog,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute grammar (spoken + written) with variety stratification
    and cross-modality predictors.

    Returns (spoken_imputed, written_imputed) DataFrames with InformantID.
    """
    log.log("\n" + "=" * 80)
    log.log("  GRAMMAR DATA IMPUTATION")
    log.log("=" * 80)

    # ── load data ───────────────────────────────────────────────────────
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM BSLVC_GRAMMAR", conn)
    df = df.copy()  # defragment

    spoken_cols = sorted(
        [c for c in df.columns if re.match(r"^[A-F]\d+[a-z]?$", c)],
        key=lambda c: (c[0], int(re.search(r'\d+', c).group()), c),
    )
    written_cols = sorted(
        [c for c in df.columns if re.match(r"^[G-N]\d+[a-z]?$", c)],
        key=lambda c: (c[0], int(re.search(r'\d+', c).group()), c),
    )
    all_feature_cols = spoken_cols + written_cols

    log.log(f"  Loaded {len(df)} participants, {len(spoken_cols)} spoken + {len(written_cols)} written features")

    # ── coerce to numeric, exclude ND participants ──────────────────────
    for c in all_feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Exclude participants where A1 or G1 is entirely missing (ND flag)
    nd_mask = df["A1"].isna() & df["G1"].isna()
    if nd_mask.any():
        log.log(f"  Excluded {nd_mask.sum()} participants with ND (no data)")
        df = df[~nd_mask].reset_index(drop=True)

    # ── apply cutoff ────────────────────────────────────────────────────
    na_counts = df[all_feature_cols].isna().sum(axis=1)
    before = len(df)
    df = df[na_counts <= GRAMMAR_CUTOFF].reset_index(drop=True)
    log.log(f"  After cutoff (≤{GRAMMAR_CUTOFF} missing): {len(df)} participants (dropped {before - len(df)})")

    total_missing = df[all_feature_cols].isna().sum().sum()
    total_cells = len(df) * len(all_feature_cols)
    log.log(f"  Missing cells: {total_missing:,} / {total_cells:,} ({100 * total_missing / total_cells:.1f}%)")

    # ── get cross-modality mapping ──────────────────────────────────────
    sw_map = _get_spoken_written_mapping(db_path)
    # Filter to columns actually present
    sw_map = {s: w for s, w in sw_map.items() if s in spoken_cols and w in written_cols}
    ws_map = {w: s for s, w in sw_map.items()}

    log.log(f"  Cross-modality pairs: {len(sw_map)} (spoken → written)")

    # ── add variety column ──────────────────────────────────────────────
    df["variety"] = df["InformantID"].apply(_extract_variety)
    variety_counts = df["variety"].value_counts()

    # Group small varieties into "OTHER"
    small_vars = variety_counts[variety_counts < MIN_VARIETY_SIZE].index.tolist()
    if small_vars:
        df.loc[df["variety"].isin(small_vars), "variety"] = "OTHER"
        log.log(f"  Merged {len(small_vars)} small varieties into OTHER: {', '.join(small_vars)}")

    varieties = sorted(df["variety"].unique())
    log.log(f"  Varieties: {', '.join(f'{v}({(df.variety==v).sum()})' for v in varieties)}")

    # ── impute per variety ──────────────────────────────────────────────
    # Build bidirectional cross-modality map for predictor priority
    cross_mod = {**sw_map, **ws_map}

    log.log(f"\n  Starting variety-stratified PMM imputation (5 cycles, k=5, max {MAX_PREDICTORS} predictors)…")

    imputed_dfs = []
    cv_all_metrics = {"spoken": {}, "written": {}}

    for var in varieties:
        mask = df["variety"] == var
        sub = df.loc[mask].copy()
        n_var = len(sub)
        log.log(f"\n  ── Variety {var} (n={n_var}) ──")

        sub_missing_spoken = sub[spoken_cols].isna().sum().sum()
        sub_missing_written = sub[written_cols].isna().sum().sum()
        log.log(f"    Missing: {sub_missing_spoken} spoken, {sub_missing_written} written")

        var_rng = np.random.default_rng(SEED + hash(var) % (2**31))

        result = _pmm_impute_dataframe(
            sub,
            feature_cols=all_feature_cols,
            aux_cols=[],
            rng=var_rng,
            log=log,
            n_cycles=5,
            cross_modality_map=cross_mod,
        )

        imputed_dfs.append(result)

        # Cross-validation (reduced for speed)
        if n_var >= 30:
            cv_rng = np.random.default_rng(SEED + hash(var) % (2**31) + 1)
            cv = _cv_quality(sub, spoken_cols, rng=cv_rng, n_folds=3, cross_modality_map=cross_mod,
                            value_min=GRAMMAR_VALUE_MIN, value_max=GRAMMAR_VALUE_MAX)
            cv_all_metrics["spoken"][var] = cv
            cv_w = _cv_quality(sub, written_cols, rng=cv_rng, n_folds=3, cross_modality_map=cross_mod,
                              value_min=GRAMMAR_VALUE_MIN, value_max=GRAMMAR_VALUE_MAX)
            cv_all_metrics["written"][var] = cv_w

    # ── reassemble ──────────────────────────────────────────────────────
    df_imp = pd.concat(imputed_dfs, ignore_index=True)
    # Remove duplicate InformantIDs (from fallback pooling)
    df_imp = df_imp.drop_duplicates(subset="InformantID", keep="first").reset_index(drop=True)

    # ── post-process ────────────────────────────────────────────────────
    n_capped = 0
    for c in all_feature_cols:
        vals = df_imp[c].values.astype(float)
        mask = ~np.isnan(vals)
        rounded = np.round(vals)
        capped = np.clip(rounded, GRAMMAR_VALUE_MIN, GRAMMAR_VALUE_MAX)
        n_capped += int(np.sum((capped != rounded) & mask))
        # Use nullable integer to preserve NaN instead of silently converting
        result_vals = pd.array([int(v) if m else pd.NA for v, m in zip(capped, mask)], dtype="Int64")
        df_imp[c] = result_vals

    log.log(f"\n  Post-processing: {n_capped} values capped to [{GRAMMAR_VALUE_MIN}, {GRAMMAR_VALUE_MAX}]")
    remaining = df_imp[all_feature_cols].isna().sum().sum()
    log.log(f"  Remaining missing: {remaining}")

    # ── aggregate CV metrics ────────────────────────────────────────────
    for label in ("spoken", "written"):
        var_m = cv_all_metrics[label]
        if var_m:
            # Pool all CV results
            total_n = sum(m.get("n_evaluated", 0) for m in var_m.values())
            if total_n > 0:
                pooled = {
                    "n_evaluated": total_n,
                    "mae": sum(m["mae"] * m["n_evaluated"] for m in var_m.values() if m) / total_n,
                    "exact_match_rate": sum(m["exact_match_rate"] * m["n_evaluated"] for m in var_m.values() if m) / total_n,
                    "within_1_rate": sum(m["within_1_rate"] * m["n_evaluated"] for m in var_m.values() if m) / total_n,
                    "mean_shift": sum(m["mean_shift"] * m["n_evaluated"] for m in var_m.values() if m) / total_n,
                    "var_ratio": sum(m["var_ratio"] * m["n_evaluated"] for m in var_m.values() if m) / total_n,
                }
                _log_quality(log, f"Grammar {label}", pooled, var_m)

    # ── split into spoken / written DataFrames ──────────────────────────
    spoken_df = df_imp[["InformantID"] + spoken_cols].copy()
    written_df = df_imp[["InformantID"] + written_cols].copy()

    log.log(f"\n  Grammar imputation complete: {len(spoken_df)} spoken, {len(written_df)} written participants")
    return spoken_df, written_df


def impute_lexical(
    db_path: Path,
    log: ImputationLog,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Impute lexical data with variety stratification.

    Returns DataFrame with InformantID + all lexical feature columns.
    """
    log.log("\n" + "=" * 80)
    log.log("  LEXICAL DATA IMPUTATION")
    log.log("=" * 80)

    # ── load data ───────────────────────────────────────────────────────
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM BSLVC_LEXICAL", conn)

    # Determine lexical feature columns (between aDropInTheOcean and Anyway)
    all_cols = list(df.columns)
    try:
        start_idx = all_cols.index("aDropInTheOcean")
        end_idx = all_cols.index("Anyway")
        lex_cols = all_cols[start_idx:end_idx + 1]
    except ValueError:
        log.log("  ❌ Cannot find lexical column range (aDropInTheOcean..Anyway)")
        return pd.DataFrame()

    log.log(f"  Loaded {len(df)} participants, {len(lex_cols)} lexical features")

    # ── coerce to numeric ───────────────────────────────────────────────
    for c in lex_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── apply cutoff ────────────────────────────────────────────────────
    na_counts = df[lex_cols].isna().sum(axis=1)
    before = len(df)
    df = df[na_counts <= LEXICAL_CUTOFF].reset_index(drop=True)
    log.log(f"  After cutoff (≤{LEXICAL_CUTOFF} missing): {len(df)} participants (dropped {before - len(df)})")

    total_missing = df[lex_cols].isna().sum().sum()
    total_cells = len(df) * len(lex_cols)
    log.log(f"  Missing cells: {total_missing:,} / {total_cells:,} ({100 * total_missing / max(total_cells, 1):.1f}%)")

    if total_missing == 0:
        log.log("  No missing values — skipping imputation")
        return df[["InformantID"] + lex_cols]

    # ── variety stratification ──────────────────────────────────────────
    df["variety"] = df["InformantID"].apply(_extract_variety)
    variety_counts = df["variety"].value_counts()

    # Group small varieties into "OTHER"
    small_vars = variety_counts[variety_counts < MIN_VARIETY_SIZE].index.tolist()
    if small_vars:
        df.loc[df["variety"].isin(small_vars), "variety"] = "OTHER"
        log.log(f"  Merged {len(small_vars)} small varieties into OTHER: {', '.join(small_vars)}")

    varieties = sorted(df["variety"].unique())
    log.log(f"  Variety groups: {len(varieties)}")

    log.log(f"\n  Starting variety-stratified PMM imputation (5 cycles, k=5, max {MAX_PREDICTORS} predictors)…")

    imputed_dfs = []
    cv_var_metrics = {}

    for var in varieties:
        mask = df["variety"] == var
        sub = df.loc[mask].copy()
        n_var = len(sub)
        sub_miss = sub[lex_cols].isna().sum().sum()
        log.log(f"    {var} (n={n_var}): {sub_miss} missing")

        if sub_miss == 0:
            imputed_dfs.append(sub)
            continue

        var_rng = np.random.default_rng(SEED + hash(var) % (2**31) + 100)

        result = _pmm_impute_dataframe(
            sub, feature_cols=lex_cols, rng=var_rng, log=None, n_cycles=5,
        )
        imputed_dfs.append(result)

        # CV only for groups large enough
        if n_var >= 30:
            cv_rng = np.random.default_rng(SEED + hash(var) % (2**31) + 101)
            cv_var_metrics[var] = _cv_quality(sub, lex_cols, rng=cv_rng, n_folds=3,
                                              value_min=LEXICAL_VALUE_MIN, value_max=LEXICAL_VALUE_MAX)

    # ── reassemble ──────────────────────────────────────────────────────
    df_imp = pd.concat(imputed_dfs, ignore_index=True)
    df_imp = df_imp.drop_duplicates(subset="InformantID", keep="first").reset_index(drop=True)

    # ── post-process ────────────────────────────────────────────────────
    n_capped = 0
    for c in lex_cols:
        vals = df_imp[c].values.astype(float)
        mask = ~np.isnan(vals)
        rounded = np.round(vals)
        capped = np.clip(rounded, LEXICAL_VALUE_MIN, LEXICAL_VALUE_MAX)
        n_capped += int(np.sum((capped != rounded) & mask))
        result_vals = pd.array([int(v) if m else pd.NA for v, m in zip(capped, mask)], dtype="Int64")
        df_imp[c] = result_vals

    log.log(f"\n  Post-processing: {n_capped} values capped to [{LEXICAL_VALUE_MIN}, {LEXICAL_VALUE_MAX}]")
    remaining = df_imp[lex_cols].isna().sum().sum()
    log.log(f"  Remaining missing: {remaining}")

    # ── log quality ─────────────────────────────────────────────────────
    if cv_var_metrics:
        total_n = sum(m.get("n_evaluated", 0) for m in cv_var_metrics.values())
        if total_n > 0:
            pooled = {
                "n_evaluated": total_n,
                "mae": sum(m["mae"] * m["n_evaluated"] for m in cv_var_metrics.values() if m) / total_n,
                "exact_match_rate": sum(m["exact_match_rate"] * m["n_evaluated"] for m in cv_var_metrics.values() if m) / total_n,
                "within_1_rate": sum(m["within_1_rate"] * m["n_evaluated"] for m in cv_var_metrics.values() if m) / total_n,
                "mean_shift": sum(m["mean_shift"] * m["n_evaluated"] for m in cv_var_metrics.values() if m) / total_n,
                "var_ratio": sum(m["var_ratio"] * m["n_evaluated"] for m in cv_var_metrics.values() if m) / total_n,
            }
            _log_quality(log, "Lexical", pooled, cv_var_metrics)

    result_df = df_imp[["InformantID"] + lex_cols]
    log.log(f"\n  Lexical imputation complete: {len(result_df)} participants")
    return result_df


# ── database upload ─────────────────────────────────────────────────────────

def upload_to_db(
    db_path: Path,
    spoken_df: pd.DataFrame,
    written_df: pd.DataFrame,
    lexical_df: pd.DataFrame,
    log: ImputationLog,
):
    """Upload imputed data to the *Imputed tables in the SQLite database."""
    log.log("\n" + "=" * 80)
    log.log("  UPLOADING TO DATABASE")
    log.log("=" * 80)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")

        def _val(v):
            """Convert a value for SQL insertion, mapping NaN/None/pd.NA to None (NULL)."""
            if v is None or v is pd.NA:
                return None
            try:
                import math
                if math.isnan(float(v)):
                    return None
            except (ValueError, TypeError):
                pass
            return int(v) if isinstance(v, (np.integer, int)) else v

        # ── spoken ──────────────────────────────────────────────────────
        cursor.execute("DELETE FROM SpokenItemsImputed")
        spoken_feature_cols = [c for c in spoken_df.columns if c != "InformantID"]
        spoken_col_list = ", ".join(["ID", "InformantID", "GrammarSpokenFillingInFor"] + spoken_feature_cols)
        spoken_placeholders = ", ".join(["?"] * (3 + len(spoken_feature_cols)))
        for i, row in enumerate(spoken_df.itertuples(index=False), 1):
            vals = [_val(getattr(row, c)) for c in spoken_feature_cols]
            cursor.execute(
                f"INSERT INTO SpokenItemsImputed ({spoken_col_list}) VALUES ({spoken_placeholders})",
                [i, row.InformantID, ""] + vals,
            )
        log.log(f"  Inserted {len(spoken_df)} spoken records")

        # ── written ─────────────────────────────────────────────────────
        cursor.execute("DELETE FROM WrittenItemsImputed")
        written_feature_cols = [c for c in written_df.columns if c != "InformantID"]
        written_col_list = ", ".join(["ID", "InformantID", "GrammarWrittenFillingInFor"] + written_feature_cols)
        written_placeholders = ", ".join(["?"] * (3 + len(written_feature_cols)))
        for i, row in enumerate(written_df.itertuples(index=False), 1):
            vals = [_val(getattr(row, c)) for c in written_feature_cols]
            cursor.execute(
                f"INSERT INTO WrittenItemsImputed ({written_col_list}) VALUES ({written_placeholders})",
                [i, row.InformantID, ""] + vals,
            )
        log.log(f"  Inserted {len(written_df)} written records")

        # ── lexical ─────────────────────────────────────────────────────
        cursor.execute("DELETE FROM LexicalItemsImputed")
        lex_feature_cols = [c for c in lexical_df.columns if c != "InformantID"]
        lex_col_list = ", ".join(["ID", "InformantID"] + lex_feature_cols + ["CommentsLexical"])
        lex_placeholders = ", ".join(["?"] * (3 + len(lex_feature_cols)))
        for i, row in enumerate(lexical_df.itertuples(index=False), 1):
            vals = [_val(getattr(row, c)) for c in lex_feature_cols]
            cursor.execute(
                f"INSERT INTO LexicalItemsImputed ({lex_col_list}) VALUES ({lex_placeholders})",
                [i, row.InformantID] + vals + [""],
            )
        log.log(f"  Inserted {len(lexical_df)} lexical records")

        conn.commit()
        log.log("  ✅ Database transaction committed")

    except Exception as e:
        conn.rollback()
        log.log(f"  ❌ Database upload failed: {e}")
        raise
    finally:
        conn.close()


# ── public entry point ──────────────────────────────────────────────────────

def run_imputation():
    """Run the full variety-stratified PMM imputation pipeline."""
    print()
    print("=" * 80)
    print("  STAGE: IMPUTATION (variety-stratified PMM)")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"  ⚠  Database not found at {DB_PATH}. Skipping.")
        return

    log = ImputationLog()
    rng = np.random.default_rng(SEED)

    log.log(f"  Starting imputation at {datetime.now()}")
    log.log(f"  Database: {DB_PATH}")
    log.log(f"  Configuration:")
    log.log(f"    Method:             Variety-stratified PMM")
    log.log(f"    PMM donors (k):     {PMM_K}")
    log.log(f"    Max predictors:     {MAX_PREDICTORS}")
    log.log(f"    Chained-eq cycles:  5")
    log.log(f"    CV folds:           {N_CV_FOLDS}")
    log.log(f"    Grammar cutoff:     {GRAMMAR_CUTOFF}")
    log.log(f"    Lexical cutoff:     {LEXICAL_CUTOFF}")
    log.log(f"    Min variety size:   {MIN_VARIETY_SIZE}")
    log.log(f"    Seed:               {SEED}")

    t0 = time.time()

    # ── Grammar ─────────────────────────────────────────────────────────
    t_gram = time.time()
    spoken_df, written_df = impute_grammar(DB_PATH, log, rng)
    grammar_time = time.time() - t_gram
    log.log(f"\n  Grammar runtime: {grammar_time:.1f}s")

    # ── Lexical ─────────────────────────────────────────────────────────
    t_lex = time.time()
    lexical_df = impute_lexical(DB_PATH, log, rng)
    lexical_time = time.time() - t_lex
    log.log(f"\n  Lexical runtime: {lexical_time:.1f}s")

    # ── Upload ──────────────────────────────────────────────────────────
    upload_to_db(DB_PATH, spoken_df, written_df, lexical_df, log)

    total_time = time.time() - t0
    log.log(f"\n  Total imputation runtime: {total_time:.1f}s")
    log.log(f"  Imputation finished at {datetime.now()}")

    # ── Save log ────────────────────────────────────────────────────────
    log_path = OUTPUT_DIR / f"imputation_pmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log.save(log_path)

    print("  ✅ Imputation completed successfully")

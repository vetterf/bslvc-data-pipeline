#!/usr/bin/env Rscript
# run_limesurvey_syntax.R
# ---------------------------------------------------------------------------
# Execute a LimeSurvey R syntax file, then export the resulting data frame
# as a plain CSV for further processing by the Python pipeline.
#
# Usage:
#   Rscript run_limesurvey_syntax.R <data_csv> <syntax_file> <output_csv>
#
# Arguments:
#   data_csv     – absolute path to the LimeSurvey R data file CSV
#                  (e.g. survey_998353_R_data_file.csv)
#   syntax_file  – absolute path to the matching R syntax file
#                  (e.g. survey_998353_R_syntax_file.R)
#   output_csv   – absolute path where the intermediate CSV should be written
#
# Notes:
#   • The syntax file reads the data CSV via a *relative* path, so this
#     script temporarily changes the working directory to the directory
#     containing the data CSV before sourcing.
#   • Grammar items (AO01→AO06) are already resolved to human-readable
#     labels ("Everyone", "Most", …) by the syntax file.  The Python
#     _GRAMMAR_VALUE_MAP handles both the raw codes and the labels, so
#     either form works.
#   • Lexical items (LIA1–LIA6) are a special case: the syntax file maps
#     LIA1/LIA5 → "Always" and LIA2/LIA4 → "More often", making the
#     direction ambiguous.  To preserve the original scale direction this
#     script restores the raw LIA codes for all lexical columns
#     (column names matching ^LI[0-9]+_SQ).
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_limesurvey_syntax.R <data_csv> <syntax_file> <output_csv>")
}

data_csv    <- normalizePath(args[1], mustWork = TRUE)
syntax_file <- normalizePath(args[2], mustWork = TRUE)
output_csv  <- args[3]   # need not exist yet

cat("  [R] data CSV   :", data_csv, "\n")
cat("  [R] syntax file:", syntax_file, "\n")
cat("  [R] output CSV :", output_csv, "\n\n")

# ── Step 1: Read raw data to preserve LIA codes ───────────────────────────
cat("  [R] Reading raw data ...\n")
raw_data <- read.csv(
  data_csv,
  quote           = "'\"",
  na.strings      = c("", "\"\""),
  stringsAsFactors = FALSE,
  fileEncoding    = "UTF-8-BOM",
  check.names     = FALSE
)
cat("  [R]", nrow(raw_data), "respondents,", ncol(raw_data), "raw columns\n")

# ── Step 2: Source the R syntax file ──────────────────────────────────────
# The syntax file uses a relative path to find the data CSV, so we must
# cd into the data directory first.
old_wd <- getwd()
setwd(dirname(data_csv))

cat("  [R] Sourcing syntax file ...\n")
# 'data' is (re-)created inside the syntax file
source(syntax_file)

setwd(old_wd)
cat("  [R] Syntax file done.", nrow(data), "rows,", ncol(data), "cols after labelling\n")

# ── Step 3: Restore raw LIA codes for lexical columns ─────────────────────
# Lexical columns have a factor whose labels collapse LIA1=LIA5="Always" and
# LIA2=LIA4="More often", so the numeric direction cannot be recovered.
# We restore the original LIA1-LIA6 codes from raw_data using the column
# position (both data frames have the same column order).
lia_colnames <- grep("^LI[0-9]+_SQ", colnames(data), value = TRUE)
if (length(lia_colnames) > 0) {
  cat("  [R] Restoring raw LIA codes for", length(lia_colnames), "lexical columns\n")
  for (col in lia_colnames) {
    col_idx        <- which(colnames(data) == col)
    data[[col]]    <- raw_data[, col_idx]
  }
}

# ── Step 4: Write intermediate CSV ────────────────────────────────────────
# Convert all factor columns to character so write.csv doesn't encode them.
for (col in colnames(data)) {
  if (is.factor(data[[col]])) {
    data[[col]] <- as.character(data[[col]])
  }
}

dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(data, file = output_csv, row.names = FALSE, na = "",
          fileEncoding = "UTF-8")

cat("  [R] Intermediate CSV written:", output_csv, "\n")
cat("  [R] Done.\n")

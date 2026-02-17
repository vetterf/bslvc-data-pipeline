#!/usr/bin/env Rscript
#####
# fabOF Chained-Forest Imputation for BSLVC Grammar and Lexical Data
#
# Uses Frequency-Adjusted Borders Ordinal Forest (fabOF; Buczak, 2025)
# in a chained-equations framework (like missForest) for ordinal data.
#
# fabOF does NOT natively support imputation — this script wraps fabOF's
# single-target prediction into an iterative column-by-column imputation
# loop with convergence monitoring via out-of-bag (OOB) prediction error.
#
# Usage:  Rscript BSLVC_imputation_fabOF.R <data_directory> [test]
#
#   <data_directory>  Path containing BSLVC_GRAMMAR.rds, BSLVC_LEXICAL.rds
#                     and BSLVC_sqlite.db
#   test              If "test", runs a quick smoke test on a small subset
#####

################################################################################
# COMMAND-LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript BSLVC_imputation_fabOF.R <data_directory> [test]")
}

DATA_DIR <- args[1]
TEST_MODE <- length(args) >= 2 && tolower(args[2]) == "test"

################################################################################
# CONFIGURATION
################################################################################

CONFIG <- list(
  # fabOF chained-forest parameters
  MAX_ITER       = 10,        # max imputation iterations
  NUM_TREES      = 100,       # number of trees per fabOF model
  CONV_THRESHOLD = 1e-4,      # relative change threshold for convergence

  # Data processing options
  USE_VARIETY_PREDICTOR     = TRUE,
  MIN_VARIETY_SIZE_GRAMMAR  = 15,
  MIN_VARIETY_SIZE_LEXICAL  = 50,

  # Cutoff thresholds
  GRAMMAR_CUTOFF = 50,
  LEXICAL_CUTOFF = 25,

  # General
  SEED = 211191,

  # Test mode: small subset
  TEST_N_ROWS    = 80,
  TEST_N_COLS    = 15,
  TEST_NUM_TREES = 50,
  TEST_MAX_ITER  = 3,

  # Database upload
  UPLOAD_TO_DB = TRUE,
  DB_PATH = file.path(DATA_DIR, "BSLVC_sqlite.db")
)

# Override for test mode
if (TEST_MODE) {
  CONFIG$NUM_TREES  <- CONFIG$TEST_NUM_TREES
  CONFIG$MAX_ITER   <- CONFIG$TEST_MAX_ITER
  CONFIG$UPLOAD_TO_DB <- FALSE
  cat("*** TEST MODE: using small subset, reduced trees, no DB upload ***\n\n")
}

# Grammar items ranges
GRAMMAR_SPOKEN_START <- "A1"
GRAMMAR_SPOKEN_END   <- "F23"
GRAMMAR_WRITTEN_START <- "G1"
GRAMMAR_WRITTEN_END   <- "N25"

# Lexical items ranges
LEXICAL_START <- "aDropInTheOcean"
LEXICAL_END   <- "Anyway"

################################################################################
# LOAD LIBRARIES
################################################################################

library(data.table)

# Install fabOF if not available
if (!requireNamespace("fabOF", quietly = TRUE)) {
  cat("fabOF not found — installing from GitHub...\n")
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  devtools::install_github("phibuc/fabOF")
}
library(fabOF)
library(ranger)    # required by fabOF

if (CONFIG$UPLOAD_TO_DB) {
  library(RSQLite)
}

set.seed(CONFIG$SEED)

################################################################################
# SETUP LOGGING
################################################################################

log_file <- file.path(DATA_DIR, paste0("imputation_fabof_",
                   format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
log_conn <- file(log_file, open = "wt")
sink(log_conn, type = "output", split = TRUE)
sink(log_conn, type = "message")

print(paste("Starting fabOF chained-forest imputation at", Sys.time()))
print(paste("Log file:", log_file))
print(paste("Data directory:", DATA_DIR))
print(paste("Test mode:", TEST_MODE))
print("")
print("=== CONFIGURATION ===")
print(paste("Max iterations:", CONFIG$MAX_ITER))
print(paste("Number of trees:", CONFIG$NUM_TREES))
print(paste("Convergence threshold:", CONFIG$CONV_THRESHOLD))
print(paste("Use variety predictor:", CONFIG$USE_VARIETY_PREDICTOR))
print(paste("Grammar cutoff:", CONFIG$GRAMMAR_CUTOFF))
print(paste("Lexical cutoff:", CONFIG$LEXICAL_CUTOFF))
print(paste("Random seed:", CONFIG$SEED))
print(paste("Upload to database:", CONFIG$UPLOAD_TO_DB))
print("")

################################################################################
# HELPER FUNCTIONS
################################################################################

# Group small varieties into "Other" category
group_small_varieties <- function(variety_values, min_size = 15, data_name = "data") {
  variety_counts <- table(variety_values, useNA = "no")
  small_varieties <- names(variety_counts[variety_counts < min_size])

  if (length(small_varieties) > 0) {
    variety_values_grouped <- ifelse(variety_values %in% small_varieties,
                                    "Other", variety_values)
    n_original <- length(unique(variety_values[!is.na(variety_values)]))
    n_final    <- length(unique(variety_values_grouped[!is.na(variety_values_grouped)]))

    print(paste("  Variety grouping for", data_name, ":"))
    print(paste("    Original varieties:", n_original))
    print(paste("    Small varieties (n <", min_size, "):", length(small_varieties)))
    print(paste("    Final variety count:", n_final))

    return(variety_values_grouped)
  } else {
    print(paste("  No varieties grouped for", data_name))
    return(variety_values)
  }
}

################################################################################
# CORE: CHAINED fabOF IMPUTATION
#
# Algorithm (analogous to missForest):
#   1. Initialise missing values with the column mode.
#   2. Sort columns by ascending missingness.
#   3. For each iteration:
#        For each column j with missing values:
#          a. Set column j as the ordinal target (factor).
#          b. Use all other columns as predictors.
#          c. Fit a fabOF on the observed rows.
#          d. Predict the missing rows and fill them in.
#   4. Check convergence via the relative change in the imputed matrix.
#   5. Return the last imputed matrix before convergence or max_iter.
################################################################################

impute_with_fabOF <- function(data_to_impute, data_name,
                              ordinal_levels,  # e.g. 0:5 or (-2):2
                              has_variety_col = FALSE,
                              num_trees = CONFIG$NUM_TREES,
                              max_iter  = CONFIG$MAX_ITER,
                              conv_threshold = CONFIG$CONV_THRESHOLD) {

  print(sprintf("\n=== RUNNING fabOF chained-forest imputation for %s ===", data_name))

  # ── Ensure we work on a plain data.frame (avoid data.table syntax issues)
  data_to_impute <- as.data.frame(data_to_impute)

  # ── Identify feature columns (everything except variety) ──────────────
  all_cols  <- names(data_to_impute)
  feat_cols <- if (has_variety_col) all_cols[all_cols != "variety"] else all_cols
  n_feat    <- length(feat_cols)
  n_obs     <- nrow(data_to_impute)

  # Convert feature columns to numeric first (for distance calcs)
  for (col in feat_cols) {
    data_to_impute[[col]] <- suppressWarnings(as.numeric(data_to_impute[[col]]))
  }

  # Create the missingness mask
  miss_mask <- is.na(as.matrix(data_to_impute[, feat_cols, drop = FALSE]))

  total_cells   <- length(miss_mask)
  missing_cells <- sum(miss_mask)
  missing_pct   <- (missing_cells / total_cells) * 100

  print(sprintf("Observations: %d, Features: %d", n_obs, n_feat))
  print(sprintf("Total cells: %s", format(total_cells, big.mark = ",")))
  print(sprintf("Missing cells: %s (%.2f%%)", format(missing_cells, big.mark = ","), missing_pct))

  if (missing_cells == 0) {
    print("No missing values to impute!")
    return(list(data_imputed = data_to_impute, runtime_mins = 0,
                oob_error = NA, n_iter = 0))
  }

  # ── Which columns have missing values? Sort by ascending missingness ─
  col_miss_count <- colSums(miss_mask)
  cols_to_impute <- names(sort(col_miss_count[col_miss_count > 0]))
  print(sprintf("Columns with missing values: %d / %d", length(cols_to_impute), n_feat))

  # ── Step 1: Initialise missing values with column mode ────────────────
  mode_fn <- function(x) {
    ux <- na.omit(x)
    if (length(ux) == 0) return(ordinal_levels[1])
    as.numeric(names(sort(table(ux), decreasing = TRUE))[1])
  }

  data_imp <- as.data.frame(data_to_impute)  # work on a plain data.frame
  for (col in cols_to_impute) {
    na_idx <- which(is.na(data_imp[[col]]))
    data_imp[na_idx, col] <- mode_fn(data_imp[[col]])
  }

  # ── Ordinal level labels (as character for factor levels) ─────────────
  level_labels <- as.character(ordinal_levels)

  # ── Iterative imputation ─────────────────────────────────────────────
  start_time <- Sys.time()
  prev_imp   <- as.matrix(data_imp[, feat_cols])  # for convergence check
  oob_errors <- numeric(0)

  for (iter in 1:max_iter) {
    print(sprintf("\n--- Iteration %d / %d ---", iter, max_iter))
    iter_errors <- c()

    for (j in seq_along(cols_to_impute)) {
      target_col <- cols_to_impute[j]

      # Rows where target was originally missing vs observed
      obs_rows  <- which(!miss_mask[, target_col])
      miss_rows <- which( miss_mask[, target_col])

      if (length(miss_rows) == 0 || length(obs_rows) < 5) next

      # Predictor columns: all features except target (+ variety if present)
      pred_cols <- feat_cols[feat_cols != target_col]
      if (has_variety_col) pred_cols <- c(pred_cols, "variety")

      # Build training data: observed rows only
      train_df <- data_imp[obs_rows, c(pred_cols, target_col), drop = FALSE]
      # Ensure target is an ordered factor with all ordinal levels
      train_df[[target_col]] <- factor(train_df[[target_col]],
                                       levels = ordinal_levels,
                                       ordered = TRUE)

      # Drop levels that are completely absent to avoid fitting errors
      # but keep the ordered structure
      present_levels <- levels(droplevels(train_df[[target_col]]))
      if (length(present_levels) <= 1) {
        # Only one value observed — fill missing with that value
        data_imp[miss_rows, target_col] <- as.numeric(present_levels[1])
        next
      }

      # Re-factor with only present levels (fabOF needs > 1 category)
      train_df[[target_col]] <- factor(train_df[[target_col]],
                                       levels = present_levels,
                                       ordered = TRUE)

      formula_str <- paste0("`", target_col, "` ~ .")
      formula_obj <- as.formula(formula_str)

      # Fit fabOF
      tryCatch({
        fit <- fabOF(formula = formula_obj,
                     data    = train_df,
                     ranger.control = list(num.trees = num_trees))

        # Predict for missing rows
        pred_df <- data_imp[miss_rows, pred_cols, drop = FALSE]
        preds   <- predict(fit, newdata = pred_df)

        # Convert factor predictions back to numeric
        data_imp[miss_rows, target_col] <- as.numeric(as.character(preds))

        # Collect OOB error from the underlying ranger fit
        # ranger's prediction.error is MSE for regression
        oob_mse <- fit$ranger.fit$prediction.error
        iter_errors <- c(iter_errors, oob_mse)

      }, error = function(e) {
        # If fabOF fails for this column (e.g. too few observations),
        # silently skip — the mode initialisation remains
        print(sprintf("  Warning: fabOF failed for column '%s': %s",
                      target_col, e$message))
      })

      # Progress indicator every 25 columns
      if (j %% 25 == 0) {
        print(sprintf("  Processed %d / %d columns", j, length(cols_to_impute)))
      }
    }

    # ── OOB error summary for this iteration ──────────────────────────
    mean_oob <- if (length(iter_errors) > 0) mean(iter_errors) else NA
    oob_errors <- c(oob_errors, mean_oob)
    print(sprintf("  Mean OOB MSE across columns: %.6f", mean_oob))

    # ── Convergence check ─────────────────────────────────────────────
    curr_imp   <- as.matrix(data_imp[, feat_cols])
    # Relative change: sum of squared differences / sum of squared values
    diff_sq    <- sum((curr_imp - prev_imp)^2, na.rm = TRUE)
    curr_sq    <- sum(curr_imp^2, na.rm = TRUE)
    rel_change <- if (curr_sq > 0) diff_sq / curr_sq else 0

    print(sprintf("  Relative change: %.8f (threshold: %.8f)",
                  rel_change, conv_threshold))

    if (iter > 1 && rel_change < conv_threshold) {
      print(sprintf("  ✓ Converged after %d iterations", iter))
      break
    }

    prev_imp <- curr_imp
  }

  end_time     <- Sys.time()
  runtime_mins <- as.numeric(difftime(end_time, start_time, units = "mins"))
  final_oob    <- if (length(oob_errors) > 0) tail(oob_errors, 1) else NA

  print(sprintf("\nfabOF chained-forest completed in %.2f minutes", runtime_mins))
  print(sprintf("Final mean OOB MSE: %.6f", final_oob))
  print(sprintf("Iterations: %d", min(iter, max_iter)))

  # Remove variety column if it was used as predictor only
  if (has_variety_col && "variety" %in% names(data_imp)) {
    data_imp$variety <- NULL
  }

  return(list(
    data_imputed = as.data.table(data_imp),
    runtime_mins = runtime_mins,
    oob_error    = final_oob,
    n_iter       = min(iter, max_iter),
    oob_history  = oob_errors
  ))
}

# Post-process: round and cap to valid ordinal range
post_process_imputed <- function(data_imputed, min_val = 0, max_val = 5) {
  print("\n=== POST-PROCESSING ===")

  n_capped <- 0
  for (col in names(data_imputed)) {
    data_imputed[[col]] <- suppressWarnings(as.numeric(data_imputed[[col]]))
    data_imputed[[col]] <- round(data_imputed[[col]])
    n_capped <- n_capped + sum(data_imputed[[col]] > max_val |
                               data_imputed[[col]] < min_val, na.rm = TRUE)
    data_imputed[[col]] <- pmin(pmax(data_imputed[[col]], min_val), max_val)
  }

  print(sprintf("Values capped to [%d, %d] range: %d", min_val, max_val, n_capped))
  return(data_imputed)
}

# Upload imputed data to database (reused from unified script)
upload_to_database <- function(grammar_imputed, lexical_imputed) {
  print("\n")
  print("================================================================================")
  print("                      UPLOADING TO DATABASE                                     ")
  print("================================================================================")
  print("\n")

  print(paste("Connecting to database:", CONFIG$DB_PATH))
  dbhandle <- dbConnect(SQLite(), dbname = CONFIG$DB_PATH)

  grc <- which(colnames(grammar_imputed) == "A1"):which(colnames(grammar_imputed) == "N25")
  lc  <- which(colnames(lexical_imputed) == "aDropInTheOcean"):which(colnames(lexical_imputed) == "Anyway")

  grammar_imputed[, (grc) := lapply(.SD, as.numeric), .SDcols = grc]
  lexical_imputed[, (lc)  := lapply(.SD, as.numeric), .SDcols = lc]

  grammar_imputed[, (grc) := lapply(.SD, function(x) round(x, 0)), .SDcols = grc]
  lexical_imputed[, (lc)  := lapply(.SD, function(x) round(x, 0)), .SDcols = lc]

  SpokenCols  <- c("InformantID", colnames(grammar_imputed)[which(colnames(grammar_imputed) == "A1"):which(colnames(grammar_imputed) == "F23")])
  WrittenCols <- c("InformantID", colnames(grammar_imputed)[which(colnames(grammar_imputed) == "G1"):which(colnames(grammar_imputed) == "N25")])
  BSLVC_Gr_Spoken  <- grammar_imputed[, ..SpokenCols]
  BSLVC_Gr_Written <- grammar_imputed[, ..WrittenCols]
  LexCols <- colnames(lexical_imputed)[!(colnames(lexical_imputed) == "InformantID")]

  LexIDs       <- dbGetQuery(dbhandle, "select InformantID from LexicalItemsImputed")
  GrSpokenIDs  <- dbGetQuery(dbhandle, "select InformantID from SpokenItemsImputed")
  GrWrittenIDs <- dbGetQuery(dbhandle, "select InformantID from WrittenItemsImputed")

  GrSpokenIDs  <- GrSpokenIDs[GrSpokenIDs$InformantID %in% BSLVC_Gr_Spoken$InformantID, ]
  GrWrittenIDs <- GrWrittenIDs[GrWrittenIDs$InformantID %in% BSLVC_Gr_Written$InformantID, ]
  LexIDs       <- LexIDs[LexIDs$InformantID %in% lexical_imputed$InformantID, ]

  print(paste("Found", length(GrSpokenIDs), "existing spoken records"))
  print(paste("Found", length(GrWrittenIDs), "existing written records"))
  print(paste("Found", length(LexIDs), "existing lexical records"))

  dbBegin(dbhandle)

  tryCatch({
    if (length(GrSpokenIDs) > 0) {
      invisible(lapply(GrSpokenIDs, function(x) {
        dbExecute(dbhandle, paste0("delete from SpokenItemsImputed where InformantID='", x, "'"))
      }))
    }
    if (length(GrWrittenIDs) > 0) {
      invisible(lapply(GrWrittenIDs, function(x) {
        dbExecute(dbhandle, paste0("delete from WrittenItemsImputed where InformantID='", x, "'"))
      }))
    }
    if (length(LexIDs) > 0) {
      invisible(lapply(LexIDs, function(x) {
        dbExecute(dbhandle, paste0("delete from LexicalItemsImputed where InformantID='", x, "'"))
      }))
    }

    SC <- SpokenCols[-1]
    invisible(lapply(1:nrow(BSLVC_Gr_Spoken), function(i) {
      dbExecute(dbhandle, paste0("insert into SpokenItemsImputed values(",
                                 i, ",'", BSLVC_Gr_Spoken$InformantID[i], "','',",
                                 paste0(BSLVC_Gr_Spoken[i, ..SC], collapse = ","), ")"))
    }))
    print(paste("Inserted", nrow(BSLVC_Gr_Spoken), "spoken records"))

    WC <- WrittenCols[-1]
    invisible(lapply(1:nrow(BSLVC_Gr_Written), function(i) {
      dbExecute(dbhandle, paste0("insert into WrittenItemsImputed values(",
                                 i, ",'", BSLVC_Gr_Written$InformantID[i], "','',",
                                 paste0(BSLVC_Gr_Written[i, ..WC], collapse = ","), ")"))
    }))
    print(paste("Inserted", nrow(BSLVC_Gr_Written), "written records"))

    LC <- LexCols
    invisible(lapply(1:nrow(lexical_imputed), function(i) {
      dbExecute(dbhandle, paste0("insert into LexicalItemsImputed values(",
                                 i, ",'", lexical_imputed$InformantID[i], "',",
                                 paste0(lexical_imputed[i, ..LC], collapse = ","), ",'')" ))
    }))
    print(paste("Inserted", nrow(lexical_imputed), "lexical records"))

    dbCommit(dbhandle)
    print("Database transaction committed successfully")
  }, error = function(e) {
    print(paste("ERROR during database upload:", e$message))
    dbRollback(dbhandle)
    stop(e)
  })

  dbDisconnect(dbhandle)
  print("Database upload completed successfully")
}

################################################################################
# GRAMMAR DATA IMPUTATION
################################################################################

print("\n")
print("================================================================================")
print("                      GRAMMAR DATA IMPUTATION (fabOF)                           ")
print("================================================================================")
print("\n")

print("=== LOADING GRAMMAR DATA ===")
grammar_data <- readRDS(file.path(DATA_DIR, "BSLVC_GRAMMAR.rds"))
print(paste("Loaded Grammar data with", nrow(grammar_data), "rows and", ncol(grammar_data), "columns"))

excludeIDs <- grammar_data[(grammar_data$A1 == "ND" | grammar_data$G1 == "ND")]$InformantID
grammar_data <- grammar_data[!grammar_data$InformantID %in% excludeIDs, ]
keepCols <- colnames(grammar_data)[!colnames(grammar_data) %in%
              c("GrammarWrittenFillingInFor", "GrammarSpokenFillingInFor")]
grammar_data <- grammar_data[, ..keepCols]

print(paste("Excluded", length(excludeIDs), "participants with ND"))
print(paste("Remaining participants:", nrow(grammar_data)))

all_cols <- colnames(grammar_data)

spoken_start_idx  <- which(all_cols == GRAMMAR_SPOKEN_START)
spoken_end_idx    <- which(all_cols == GRAMMAR_SPOKEN_END)
written_start_idx <- which(all_cols == GRAMMAR_WRITTEN_START)
written_end_idx   <- which(all_cols == GRAMMAR_WRITTEN_END)

grammar_spoken_items  <- all_cols[spoken_start_idx:spoken_end_idx]
grammar_written_items <- all_cols[written_start_idx:written_end_idx]
grammar_all_items     <- c(grammar_spoken_items, grammar_written_items)

print(sprintf("Spoken items: %d, Written items: %d, Total: %d",
              length(grammar_spoken_items), length(grammar_written_items),
              length(grammar_all_items)))

# Apply cutoff
print(sprintf("\n=== APPLYING CUTOFF (%d missing values) ===", CONFIG$GRAMMAR_CUTOFF))

for (col in grammar_all_items) {
  grammar_data[grammar_data[[col]] == "NA", (col) := NA]
}
grammar_data$NA_count <- rowSums(is.na(grammar_data[, grammar_all_items, with = FALSE]))
grammar_filtered <- grammar_data[grammar_data$NA_count <= CONFIG$GRAMMAR_CUTOFF, ]
grammar_filtered$NA_count <- NULL
informant_ids_grammar <- grammar_filtered$InformantID

print(sprintf("Filtered to %d participants", nrow(grammar_filtered)))

# Test mode: take small subset
if (TEST_MODE) {
  n_test_rows <- min(CONFIG$TEST_N_ROWS, nrow(grammar_filtered))
  n_test_cols <- min(CONFIG$TEST_N_COLS, length(grammar_all_items))
  grammar_filtered <- grammar_filtered[1:n_test_rows, ]
  informant_ids_grammar <- grammar_filtered$InformantID
  grammar_all_items <- grammar_all_items[1:n_test_cols]
  print(sprintf("TEST MODE: using %d rows × %d columns", n_test_rows, n_test_cols))
}

grammar_all_data <- grammar_filtered[, ..grammar_all_items]

# Add variety as predictor
has_variety <- FALSE
if (CONFIG$USE_VARIETY_PREDICTOR) {
  if ("MainVariety" %in% colnames(grammar_filtered)) {
    variety_values <- grammar_filtered$MainVariety
  } else {
    variety_values <- toupper(gsub("^([A-Za-z]+).*", "\\1", informant_ids_grammar))
  }
  variety_values <- group_small_varieties(variety_values,
                                          min_size = CONFIG$MIN_VARIETY_SIZE_GRAMMAR,
                                          data_name = "Grammar")
  grammar_all_data$variety <- as.factor(variety_values)
  has_variety <- TRUE
  print(paste("Added variety predictor with",
              length(unique(variety_values)), "unique varieties"))
}

# Run fabOF imputation — grammar uses 0-5 ordinal scale
grammar_result <- impute_with_fabOF(
  data_to_impute = grammar_all_data,
  data_name      = "Grammar",
  ordinal_levels = 0:5,
  has_variety_col = has_variety,
  num_trees      = CONFIG$NUM_TREES,
  max_iter       = CONFIG$MAX_ITER
)

grammar_imputed <- post_process_imputed(grammar_result$data_imputed, min_val = 0, max_val = 5)
grammar_imputed$InformantID <- informant_ids_grammar

grammar_runtime <- grammar_result$runtime_mins
grammar_oob     <- grammar_result$oob_error

print("\n=== SAVING GRAMMAR RESULTS ===")
if (!TEST_MODE) {
  output_file <- file.path(DATA_DIR, "BSLVC_GRAMMAR_IMPUTED.rds")
  saveRDS(grammar_imputed, output_file)
  print(paste("Grammar imputed data saved to:", output_file))
}

print(sprintf("Grammar imputation summary:"))
print(sprintf("  Runtime: %.2f minutes", grammar_runtime))
print(sprintf("  Iterations: %d", grammar_result$n_iter))
print(sprintf("  Final OOB MSE: %.6f", grammar_oob))
print(sprintf("  OOB history: %s",
              paste(sprintf("%.6f", grammar_result$oob_history), collapse = " → ")))

################################################################################
# LEXICAL DATA IMPUTATION
################################################################################

print("\n")
print("================================================================================")
print("                      LEXICAL DATA IMPUTATION (fabOF)                           ")
print("================================================================================")
print("\n")

print("=== LOADING LEXICAL DATA ===")
lexical_data <- readRDS(file.path(DATA_DIR, "BSLVC_LEXICAL.rds"))
print(paste("Loaded Lexical data with", nrow(lexical_data), "rows and", ncol(lexical_data), "columns"))

all_cols_lex <- colnames(lexical_data)
lex_start_idx <- which(all_cols_lex == LEXICAL_START)
lex_end_idx   <- which(all_cols_lex == LEXICAL_END)
lexical_all_items <- all_cols_lex[lex_start_idx:lex_end_idx]

print(sprintf("Lexical items: %d", length(lexical_all_items)))

# Apply cutoff
print(sprintf("\n=== APPLYING CUTOFF (%d missing values) ===", CONFIG$LEXICAL_CUTOFF))

for (col in lexical_all_items) {
  lexical_data[lexical_data[[col]] == "NA", (col) := NA]
}
lexical_data$NA_count <- rowSums(is.na(lexical_data[, lexical_all_items, with = FALSE]))
lexical_filtered <- lexical_data[lexical_data$NA_count <= CONFIG$LEXICAL_CUTOFF, ]
lexical_filtered$NA_count <- NULL
informant_ids_lexical <- lexical_filtered$InformantID

print(sprintf("Filtered to %d participants", nrow(lexical_filtered)))

# Test mode: take small subset
if (TEST_MODE) {
  n_test_rows <- min(CONFIG$TEST_N_ROWS, nrow(lexical_filtered))
  n_test_cols <- min(CONFIG$TEST_N_COLS, length(lexical_all_items))
  lexical_filtered <- lexical_filtered[1:n_test_rows, ]
  informant_ids_lexical <- lexical_filtered$InformantID
  lexical_all_items <- lexical_all_items[1:n_test_cols]
  print(sprintf("TEST MODE: using %d rows × %d columns", n_test_rows, n_test_cols))
}

lexical_all_data <- lexical_filtered[, ..lexical_all_items]

# Convert to numeric
for (col in lexical_all_items) {
  lexical_all_data[[col]] <- suppressWarnings(as.numeric(lexical_all_data[[col]]))
}

# Add variety as predictor
has_variety_lex <- FALSE
if (CONFIG$USE_VARIETY_PREDICTOR) {
  if ("MainVariety" %in% colnames(lexical_filtered)) {
    variety_values <- lexical_filtered$MainVariety
  } else {
    variety_values <- toupper(gsub("^([A-Za-z]+).*", "\\1", informant_ids_lexical))
  }
  variety_values <- group_small_varieties(variety_values,
                                          min_size = CONFIG$MIN_VARIETY_SIZE_LEXICAL,
                                          data_name = "Lexical")
  lexical_all_data$variety <- as.factor(variety_values)
  has_variety_lex <- TRUE
}

# Run fabOF imputation — lexical uses -2 to +2 ordinal scale
lexical_result <- impute_with_fabOF(
  data_to_impute = lexical_all_data,
  data_name      = "Lexical",
  ordinal_levels = (-2):2,
  has_variety_col = has_variety_lex,
  num_trees      = CONFIG$NUM_TREES,
  max_iter       = CONFIG$MAX_ITER
)

lexical_imputed <- post_process_imputed(lexical_result$data_imputed, min_val = -2, max_val = 2)
lexical_imputed$InformantID <- informant_ids_lexical

lexical_runtime <- lexical_result$runtime_mins
lexical_oob     <- lexical_result$oob_error

print("\n=== SAVING LEXICAL RESULTS ===")
if (!TEST_MODE) {
  output_file_lex <- file.path(DATA_DIR, "BSLVC_LEXICAL_IMPUTED.rds")
  saveRDS(lexical_imputed, output_file_lex)
  print(paste("Lexical imputed data saved to:", output_file_lex))
}

print(sprintf("Lexical imputation summary:"))
print(sprintf("  Runtime: %.2f minutes", lexical_runtime))
print(sprintf("  Iterations: %d", lexical_result$n_iter))
print(sprintf("  Final OOB MSE: %.6f", lexical_oob))
print(sprintf("  OOB history: %s",
              paste(sprintf("%.6f", lexical_result$oob_history), collapse = " → ")))

################################################################################
# FINAL SUMMARY
################################################################################

print("\n")
print("================================================================================")
print("               fabOF CHAINED-FOREST IMPUTATION – FINAL SUMMARY                 ")
print("================================================================================")
print("\n")
print("CONFIGURATION:")
print(sprintf("  Method: fabOF (Frequency-Adjusted Borders Ordinal Forest)"))
print(sprintf("  Max iterations: %d", CONFIG$MAX_ITER))
print(sprintf("  Trees per model: %d", CONFIG$NUM_TREES))
print(sprintf("  Convergence threshold: %g", CONFIG$CONV_THRESHOLD))
print(sprintf("  Use variety predictor: %s", CONFIG$USE_VARIETY_PREDICTOR))
print("")
print("GRAMMAR IMPUTATION:")
print(sprintf("  Participants: %d", nrow(grammar_imputed) - 1))
print(sprintf("  Items: %d", length(grammar_all_items)))
print(sprintf("  Iterations: %d", grammar_result$n_iter))
print(sprintf("  Runtime: %.2f minutes", grammar_runtime))
print(sprintf("  Final OOB MSE: %.6f", grammar_oob))
print("")
print("LEXICAL IMPUTATION:")
print(sprintf("  Participants: %d", nrow(lexical_imputed) - 1))
print(sprintf("  Items: %d", length(lexical_all_items)))
print(sprintf("  Iterations: %d", lexical_result$n_iter))
print(sprintf("  Runtime: %.2f minutes", lexical_runtime))
print(sprintf("  Final OOB MSE: %.6f", lexical_oob))
print("")
print("TOTAL:")
print(sprintf("  Total runtime: %.2f minutes", grammar_runtime + lexical_runtime))
print("")
print("Method Notes:")
print("- fabOF treats each variable as ordinal (factor with ordered levels)")
print("- Uses frequency-adjusted category borders (no optimization needed)")
print("- OOB error is the mean MSE from the underlying ranger regression forest")
print("- Convergence is checked via relative change between iterations")
print("================================================================================\n")

################################################################################
# UPLOAD TO DATABASE
################################################################################

if (CONFIG$UPLOAD_TO_DB && !TEST_MODE) {
  tryCatch({
    upload_to_database(grammar_imputed, lexical_imputed)
  }, error = function(e) {
    print(paste("ERROR: Database upload failed:", e$message))
    print("Imputation completed, but database upload was not successful")
  })
} else if (TEST_MODE) {
  print("\nDatabase upload skipped (test mode)")
} else {
  print("\nDatabase upload skipped (UPLOAD_TO_DB = FALSE)")
}

print(paste("\nImputation finished at", Sys.time()))

# Close logging
sink(type = "output")
sink(type = "message")
close(log_conn)
print(paste("Log saved to:", log_file))

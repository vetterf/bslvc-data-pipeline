#!/usr/bin/env Rscript
#####
# Unified Imputation Script for BSLVC Grammar and Lexical Data
# Supports both missForest and hdImpute methods
#
# Usage:  Rscript BSLVC_imputation_unified.R <data_directory>
#####

################################################################################
# COMMAND-LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript BSLVC_imputation_unified.R <data_directory>")
}

DATA_DIR <- args[1]

################################################################################
# CONFIGURATION
################################################################################

CONFIG <- list(
  # Imputation method
  METHOD = "hdImpute",           # Options: "missForest" or "hdImpute"

  # Data processing options
  SEPARATE_SPOKEN_WRITTEN = TRUE, # If TRUE, impute spoken and written separately (grammar only)

  # Cutoff thresholds (number of missing values per participant)
  GRAMMAR_CUTOFF = 50,
  LEXICAL_CUTOFF = 25,

  # missForest parameters
  MISSFOREST_PARALLEL_CORES = 6,
  MISSFOREST_VERBOSE = TRUE,

  # hdImpute parameters
  HDIMPUTE_PMM_K = 5,
  HDIMPUTE_BATCH_GRAMMAR = 20,
  HDIMPUTE_BATCH_LEXICAL = 10,

  # General parameters
  SEED = 211191,

  # Output options
  OUTPUT_SUFFIX = "",

  # Database upload
  UPLOAD_TO_DB = TRUE,
  DB_PATH = file.path(DATA_DIR, "BSLVC_sqlite.db")
)

# Grammar items ranges
GRAMMAR_SPOKEN_START <- "A1"
GRAMMAR_SPOKEN_END <- "F23"
GRAMMAR_WRITTEN_START <- "G1"
GRAMMAR_WRITTEN_END <- "N25"

# Lexical items ranges
LEXICAL_START <- "aDropInTheOcean"
LEXICAL_END <- "Anyway"

################################################################################
# LOAD LIBRARIES
################################################################################

library(data.table)
library(ggplot2)

if (CONFIG$METHOD == "missForest") {
  library(missForest)
  library(doParallel)
} else if (CONFIG$METHOD == "hdImpute") {
  library(hdImpute)
} else {
  stop("Invalid METHOD in CONFIG. Must be 'missForest' or 'hdImpute'")
}

if (CONFIG$UPLOAD_TO_DB) {
  library(RSQLite)
}

set.seed(CONFIG$SEED)

################################################################################
# SETUP LOGGING
################################################################################

method_suffix <- tolower(CONFIG$METHOD)
log_file <- file.path(DATA_DIR, paste0("imputation_", method_suffix, "_",
                   format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
log_conn <- file(log_file, open = "wt")
sink(log_conn, type = "output", split = TRUE)
sink(log_conn, type = "message")

print(paste("Starting imputation at", Sys.time()))
print(paste("Log file:", log_file))
print(paste("Data directory:", DATA_DIR))
print("")
print("=== CONFIGURATION ===")
print(paste("Method:", CONFIG$METHOD))
print(paste("Separate spoken/written:", CONFIG$SEPARATE_SPOKEN_WRITTEN))
print(paste("Grammar cutoff:", CONFIG$GRAMMAR_CUTOFF))
print(paste("Lexical cutoff:", CONFIG$LEXICAL_CUTOFF))
if (CONFIG$METHOD == "missForest") {
  print(paste("Parallel cores:", CONFIG$MISSFOREST_PARALLEL_CORES))
} else {
  print(paste("PMM k-neighbors:", CONFIG$HDIMPUTE_PMM_K))
  print(paste("Batch size (grammar):", CONFIG$HDIMPUTE_BATCH_GRAMMAR))
  print(paste("Batch size (lexical):", CONFIG$HDIMPUTE_BATCH_LEXICAL))
}
print(paste("Random seed:", CONFIG$SEED))
print(paste("Upload to database:", CONFIG$UPLOAD_TO_DB))
if (CONFIG$UPLOAD_TO_DB) {
  print(paste("Database path:", CONFIG$DB_PATH))
}
print("")

################################################################################
# HELPER FUNCTIONS
################################################################################

# Function to impute data using missForest
impute_with_missForest <- function(data_to_impute, data_name, treat_as_factors = TRUE) {
  print(sprintf("\n=== RUNNING missForest for %s ===", data_name))

  registerDoParallel(cores = CONFIG$MISSFOREST_PARALLEL_CORES)

  if (treat_as_factors) {
    data_prep <- as.data.table(lapply(data_to_impute, as.factor))
  } else {
    data_prep <- data_to_impute
  }

  start_time <- Sys.time()

  imputed_result <- missForest(
    data_prep,
    verbose = CONFIG$MISSFOREST_VERBOSE,
    parallelize = "forests"
  )

  end_time <- Sys.time()
  runtime_mins <- as.numeric(difftime(end_time, start_time, units = "mins"))

  print(sprintf("missForest completed in %.2f minutes", runtime_mins))
  print(sprintf("OOB error: %.4f", imputed_result$OOBerror))

  data_imputed <- imputed_result$ximp

  if (treat_as_factors) {
    data_imputed <- as.data.table(lapply(data_imputed, function(x) as.numeric(x) - 1))
  }

  return(list(
    data_imputed = data_imputed,
    runtime_mins = runtime_mins,
    oob_error = imputed_result$OOBerror
  ))
}

# Function to impute data using hdImpute
impute_with_hdImpute <- function(data_to_impute, data_name, batch_size) {
  print(sprintf("\n=== RUNNING hdImpute for %s ===", data_name))
  print("Note: hdImpute does not provide OOB errors like missForest")

  total_cells <- nrow(data_to_impute) * ncol(data_to_impute)
  missing_cells <- sum(is.na(data_to_impute))
  missing_pct <- (missing_cells / total_cells) * 100

  print(sprintf("Total cells: %s", format(total_cells, big.mark = ",")))
  print(sprintf("Missing cells: %s", format(missing_cells, big.mark = ",")))
  print(sprintf("Missing percentage: %.2f%%", missing_pct))

  if (missing_cells == 0) {
    print("No missing values to impute!")
    return(list(
      data_imputed = data_to_impute,
      runtime_mins = 0,
      oob_error = NA
    ))
  }

  start_time <- Sys.time()

  imputed_result <- hdImpute(
    data = data_to_impute,
    pmm_k = CONFIG$HDIMPUTE_PMM_K,
    batch = batch_size,
    seed = CONFIG$SEED
  )

  end_time <- Sys.time()
  runtime_mins <- as.numeric(difftime(end_time, start_time, units = "mins"))

  print(sprintf("hdImpute completed in %.2f minutes", runtime_mins))

  remaining_missing <- sum(is.na(imputed_result))
  if (remaining_missing > 0) {
    print(sprintf("WARNING: %d values were not imputed!", remaining_missing))
  } else {
    print("SUCCESS: All missing values were imputed")
  }

  return(list(
    data_imputed = as.data.table(imputed_result),
    runtime_mins = runtime_mins,
    oob_error = NA
  ))
}

# Function to post-process imputed data (round and cap values)
post_process_imputed <- function(data_imputed, min_val = 0, max_val = 5) {
  print("\n=== POST-PROCESSING ===")

  for (col in names(data_imputed)) {
    data_imputed[[col]] <- round(data_imputed[[col]])
  }

  n_capped <- 0
  for (col in names(data_imputed)) {
    n_capped <- n_capped + sum(data_imputed[[col]] > max_val | data_imputed[[col]] < min_val, na.rm = TRUE)
    data_imputed[[col]] <- pmin(pmax(data_imputed[[col]], min_val), max_val)
  }

  print(sprintf("Number of values capped to [%d, %d] range: %d", min_val, max_val, n_capped))

  return(data_imputed)
}

# Function to upload imputed data to database
upload_to_database <- function(grammar_imputed, lexical_imputed) {
  print("\n")
  print("================================================================================")
  print("                      UPLOADING TO DATABASE                                     ")
  print("================================================================================")
  print("\n")

  print(paste("Connecting to database:", CONFIG$DB_PATH))
  dbhandle <- dbConnect(SQLite(), dbname = CONFIG$DB_PATH)

  grc <- which(colnames(grammar_imputed) == "A1"):which(colnames(grammar_imputed) == "N25")
  lc <- which(colnames(lexical_imputed) == "aDropInTheOcean"):which(colnames(lexical_imputed) == "Anyway")

  grammar_imputed[, (grc) := lapply(.SD, as.numeric), .SDcols = grc]
  lexical_imputed[, (lc) := lapply(.SD, as.numeric), .SDcols = lc]

  grammar_imputed[, (grc) := lapply(.SD, function(x) round(x, 0)), .SDcols = grc]
  lexical_imputed[, (lc) := lapply(.SD, function(x) round(x, 0)), .SDcols = lc]

  SpokenCols <- c('InformantID', colnames(grammar_imputed)[which(colnames(grammar_imputed) == "A1"):which(colnames(grammar_imputed) == "F23")])
  WrittenCols <- c('InformantID', colnames(grammar_imputed)[which(colnames(grammar_imputed) == "G1"):which(colnames(grammar_imputed) == "N25")])
  BSLVC_Gr_Spoken <- grammar_imputed[, ..SpokenCols]
  BSLVC_Gr_Written <- grammar_imputed[, ..WrittenCols]
  LexCols <- colnames(lexical_imputed)[!(colnames(lexical_imputed) == "InformantID")]

  LexIDs <- dbGetQuery(dbhandle, "select InformantID from LexicalItemsImputed")
  GrSpokenIDs <- dbGetQuery(dbhandle, "select InformantID from SpokenItemsImputed")
  GrWrittenIDs <- dbGetQuery(dbhandle, "select InformantID from WrittenItemsImputed")

  GrSpokenIDs <- GrSpokenIDs[GrSpokenIDs$InformantID %in% BSLVC_Gr_Spoken$InformantID, ]
  GrWrittenIDs <- GrWrittenIDs[GrWrittenIDs$InformantID %in% BSLVC_Gr_Written$InformantID, ]
  LexIDs <- LexIDs[LexIDs$InformantID %in% lexical_imputed$InformantID, ]

  print(paste("Found", length(GrSpokenIDs), "existing spoken records to update"))
  print(paste("Found", length(GrWrittenIDs), "existing written records to update"))
  print(paste("Found", length(LexIDs), "existing lexical records to update"))

  dbBegin(dbhandle)

  tryCatch({
    if (length(GrSpokenIDs) > 0) {
      invisible(lapply(GrSpokenIDs, function(x) {
        sqlCode <- paste0("delete from SpokenItemsImputed where InformantID='", x, "'")
        dbExecute(dbhandle, sqlCode)
      }))
      print(paste("Deleted", length(GrSpokenIDs), "spoken records"))
    }

    if (length(GrWrittenIDs) > 0) {
      invisible(lapply(GrWrittenIDs, function(x) {
        sqlCode <- paste0("delete from WrittenItemsImputed where InformantID='", x, "'")
        dbExecute(dbhandle, sqlCode)
      }))
      print(paste("Deleted", length(GrWrittenIDs), "written records"))
    }

    if (length(LexIDs) > 0) {
      invisible(lapply(LexIDs, function(x) {
        sqlCode <- paste0("delete from LexicalItemsImputed where InformantID='", x, "'")
        dbExecute(dbhandle, sqlCode)
      }))
      print(paste("Deleted", length(LexIDs), "lexical records"))
    }

    print("\nInserting new records...")

    SC <- SpokenCols[-1]
    invisible(lapply(1:nrow(BSLVC_Gr_Spoken), function(i) {
      sqlCode <- paste0("insert into SpokenItemsImputed values(", i, ",'", BSLVC_Gr_Spoken$InformantID[i], "','',", paste0(BSLVC_Gr_Spoken[i, ..SC], collapse = ","), ")")
      dbExecute(dbhandle, sqlCode)
    }))
    print(paste("Inserted", nrow(BSLVC_Gr_Spoken), "spoken records"))

    WC <- WrittenCols[-1]
    invisible(lapply(1:nrow(BSLVC_Gr_Written), function(i) {
      sqlCode <- paste0("insert into WrittenItemsImputed values(", i, ",'", BSLVC_Gr_Written$InformantID[i], "','',", paste0(BSLVC_Gr_Written[i, ..WC], collapse = ","), ")")
      dbExecute(dbhandle, sqlCode)
    }))
    print(paste("Inserted", nrow(BSLVC_Gr_Written), "written records"))

    LC <- LexCols
    invisible(lapply(1:nrow(lexical_imputed), function(i) {
      sqlCode <- paste0("insert into LexicalItemsImputed values(", i, ",'", lexical_imputed$InformantID[i], "',", paste0(lexical_imputed[i, ..LC], collapse = ","), ",'')")
      dbExecute(dbhandle, sqlCode)
    }))
    print(paste("Inserted", nrow(lexical_imputed), "lexical records"))

    dbCommit(dbhandle)
    print("\nDatabase transaction committed successfully")

  }, error = function(e) {
    print(paste("ERROR during database upload:", e$message))
    dbRollback(dbhandle)
    print("Database transaction rolled back")
    stop(e)
  })

  dbDisconnect(dbhandle)
  print("Database connection closed")
  print("\nDatabase upload completed successfully")
}

################################################################################
# GRAMMAR DATA IMPUTATION
################################################################################

print("\n")
print("================================================================================")
print("                      GRAMMAR DATA IMPUTATION                                   ")
print("================================================================================")
print("\n")

print("=== LOADING GRAMMAR DATA ===")
grammar_data <- readRDS(file.path(DATA_DIR, "BSLVC_GRAMMAR.rds"))
print(paste("Loaded Grammar data with", nrow(grammar_data), "rows and", ncol(grammar_data), "columns"))

excludeIDs <- grammar_data[(grammar_data$A1 == "ND" | grammar_data$G1 == "ND")]$InformantID
grammar_data <- grammar_data[!grammar_data$InformantID %in% excludeIDs, ]
keepCols <- colnames(grammar_data)[!colnames(grammar_data) %in% c("GrammarWrittenFillingInFor", "GrammarSpokenFillingInFor")]
grammar_data <- grammar_data[, ..keepCols]

print(paste("Excluded", length(excludeIDs), "participants with ND for A1 or G1"))
print(paste("Remaining participants:", nrow(grammar_data)))

all_cols <- colnames(grammar_data)

if (GRAMMAR_SPOKEN_START %in% all_cols && GRAMMAR_WRITTEN_END %in% all_cols) {
  spoken_start_idx <- which(all_cols == GRAMMAR_SPOKEN_START)
  spoken_end_idx <- which(all_cols == GRAMMAR_SPOKEN_END)
  written_start_idx <- which(all_cols == GRAMMAR_WRITTEN_START)
  written_end_idx <- which(all_cols == GRAMMAR_WRITTEN_END)

  grammar_spoken_items <- all_cols[spoken_start_idx:spoken_end_idx]
  grammar_written_items <- all_cols[written_start_idx:written_end_idx]
  grammar_all_items <- c(grammar_spoken_items, grammar_written_items)
} else {
  stop("Cannot find required grammar columns!")
}

print(sprintf("Spoken items (%s to %s): %d", GRAMMAR_SPOKEN_START, GRAMMAR_SPOKEN_END, length(grammar_spoken_items)))
print(sprintf("Written items (%s to %s): %d", GRAMMAR_WRITTEN_START, GRAMMAR_WRITTEN_END, length(grammar_written_items)))
print(sprintf("Total grammar items: %d", length(grammar_all_items)))

print(sprintf("\n=== APPLYING CUTOFF (%d missing values) ===", CONFIG$GRAMMAR_CUTOFF))

for (col in grammar_all_items) {
  grammar_data[grammar_data[[col]] == "NA", (col) := NA]
}

grammar_data$NA_count <- rowSums(is.na(grammar_data[, grammar_all_items, with = FALSE]))

print(paste("Participants with <=", CONFIG$GRAMMAR_CUTOFF, "missing values:", sum(grammar_data$NA_count <= CONFIG$GRAMMAR_CUTOFF)))
print(paste("Participants with >", CONFIG$GRAMMAR_CUTOFF, "missing values:", sum(grammar_data$NA_count > CONFIG$GRAMMAR_CUTOFF)))

grammar_filtered <- grammar_data[grammar_data$NA_count <= CONFIG$GRAMMAR_CUTOFF, ]
print(sprintf("Filtered to %d participants", nrow(grammar_filtered)))

grammar_filtered$NA_count <- NULL
informant_ids_grammar <- grammar_filtered$InformantID

if (CONFIG$SEPARATE_SPOKEN_WRITTEN) {
  print("\n=== IMPUTING SPOKEN AND WRITTEN SEPARATELY ===")

  grammar_spoken_data <- grammar_filtered[, ..grammar_spoken_items]
  if (CONFIG$METHOD == "hdImpute") {
    for (col in grammar_spoken_items) {
      grammar_spoken_data[[col]] <- as.numeric(grammar_spoken_data[[col]])
    }
  }

  grammar_written_data <- grammar_filtered[, ..grammar_written_items]
  if (CONFIG$METHOD == "hdImpute") {
    for (col in grammar_written_items) {
      grammar_written_data[[col]] <- as.numeric(grammar_written_data[[col]])
    }
  }

  if (CONFIG$METHOD == "missForest") {
    spoken_result <- impute_with_missForest(grammar_spoken_data, "Grammar Spoken", treat_as_factors = TRUE)
  } else {
    spoken_result <- impute_with_hdImpute(grammar_spoken_data, "Grammar Spoken", CONFIG$HDIMPUTE_BATCH_GRAMMAR)
  }
  grammar_spoken_imputed <- post_process_imputed(spoken_result$data_imputed)

  if (CONFIG$METHOD == "missForest") {
    written_result <- impute_with_missForest(grammar_written_data, "Grammar Written", treat_as_factors = TRUE)
  } else {
    written_result <- impute_with_hdImpute(grammar_written_data, "Grammar Written", CONFIG$HDIMPUTE_BATCH_GRAMMAR)
  }
  grammar_written_imputed <- post_process_imputed(written_result$data_imputed)

  grammar_imputed <- cbind(grammar_spoken_imputed, grammar_written_imputed)
  grammar_imputed$InformantID <- informant_ids_grammar

  grammar_runtime <- spoken_result$runtime_mins + written_result$runtime_mins
  grammar_oob <- NA

} else {
  print("\n=== IMPUTING ALL GRAMMAR ITEMS TOGETHER ===")

  grammar_all_data <- grammar_filtered[, ..grammar_all_items]
  if (CONFIG$METHOD == "hdImpute") {
    for (col in grammar_all_items) {
      grammar_all_data[[col]] <- as.numeric(grammar_all_data[[col]])
    }
  }

  if (CONFIG$METHOD == "missForest") {
    all_result <- impute_with_missForest(grammar_all_data, "Grammar All", treat_as_factors = TRUE)
  } else {
    all_result <- impute_with_hdImpute(grammar_all_data, "Grammar All", CONFIG$HDIMPUTE_BATCH_GRAMMAR)
  }
  grammar_imputed <- post_process_imputed(all_result$data_imputed)
  grammar_imputed$InformantID <- informant_ids_grammar

  grammar_runtime <- all_result$runtime_mins
  grammar_oob <- all_result$oob_error
}

print("\n=== SAVING GRAMMAR RESULTS ===")
output_suffix <- ifelse(CONFIG$OUTPUT_SUFFIX == "", "", paste0("_", CONFIG$OUTPUT_SUFFIX))
output_file <- file.path(DATA_DIR, paste0("BSLVC_GRAMMAR_IMPUTED", output_suffix, ".rds"))
saveRDS(grammar_imputed, output_file)
print(paste("Grammar imputed data saved to:", output_file))

print(sprintf("\nGrammar imputation summary:"))
print(sprintf("  Runtime: %.2f minutes", grammar_runtime))
if (!is.na(grammar_oob)) {
  print(sprintf("  OOB error: %.4f", grammar_oob))
}

################################################################################
# LEXICAL DATA IMPUTATION
################################################################################

print("\n")
print("================================================================================")
print("                      LEXICAL DATA IMPUTATION                                   ")
print("================================================================================")
print("\n")

print("=== LOADING LEXICAL DATA ===")
lexical_data <- readRDS(file.path(DATA_DIR, "BSLVC_LEXICAL.rds"))
print(paste("Loaded Lexical data with", nrow(lexical_data), "rows and", ncol(lexical_data), "columns"))

all_cols_lex <- colnames(lexical_data)

if (LEXICAL_START %in% all_cols_lex && LEXICAL_END %in% all_cols_lex) {
  lex_start_idx <- which(all_cols_lex == LEXICAL_START)
  lex_end_idx <- which(all_cols_lex == LEXICAL_END)
  lexical_all_items <- all_cols_lex[lex_start_idx:lex_end_idx]
} else {
  stop("Cannot find required lexical columns!")
}

print(sprintf("Lexical items (%s to %s): %d", LEXICAL_START, LEXICAL_END, length(lexical_all_items)))

print(sprintf("\n=== APPLYING CUTOFF (%d missing values) ===", CONFIG$LEXICAL_CUTOFF))

for (col in lexical_all_items) {
  lexical_data[lexical_data[[col]] == "NA", (col) := NA]
}

lexical_data$NA_count <- rowSums(is.na(lexical_data[, lexical_all_items, with = FALSE]))

print(paste("Participants with <=", CONFIG$LEXICAL_CUTOFF, "missing values:", sum(lexical_data$NA_count <= CONFIG$LEXICAL_CUTOFF)))
print(paste("Participants with >", CONFIG$LEXICAL_CUTOFF, "missing values:", sum(lexical_data$NA_count > CONFIG$LEXICAL_CUTOFF)))

lexical_filtered <- lexical_data[lexical_data$NA_count <= CONFIG$LEXICAL_CUTOFF, ]
print(sprintf("Filtered to %d participants", nrow(lexical_filtered)))

lexical_filtered$NA_count <- NULL
informant_ids_lexical <- lexical_filtered$InformantID

print("\n=== IMPUTING LEXICAL DATA ===")

lexical_all_data <- lexical_filtered[, ..lexical_all_items]
if (CONFIG$METHOD == "hdImpute") {
  for (col in lexical_all_items) {
    lexical_all_data[[col]] <- as.numeric(lexical_all_data[[col]])
  }
}

if (CONFIG$METHOD == "missForest") {
  lexical_result <- impute_with_missForest(lexical_all_data, "Lexical", treat_as_factors = FALSE)
} else {
  lexical_result <- impute_with_hdImpute(lexical_all_data, "Lexical", CONFIG$HDIMPUTE_BATCH_LEXICAL)
}
lexical_imputed <- post_process_imputed(lexical_result$data_imputed)
lexical_imputed$InformantID <- informant_ids_lexical

lexical_runtime <- lexical_result$runtime_mins
lexical_oob <- lexical_result$oob_error

print("\n=== SAVING LEXICAL RESULTS ===")
output_file_lex <- file.path(DATA_DIR, paste0("BSLVC_LEXICAL_IMPUTED", output_suffix, ".rds"))
saveRDS(lexical_imputed, output_file_lex)
print(paste("Lexical imputed data saved to:", output_file_lex))

print(sprintf("\nLexical imputation summary:"))
print(sprintf("  Runtime: %.2f minutes", lexical_runtime))
if (!is.na(lexical_oob)) {
  print(sprintf("  OOB error: %.4f", lexical_oob))
}

################################################################################
# FINAL SUMMARY
################################################################################

print("\n")
print("================================================================================")
print("                          IMPUTATION FINAL SUMMARY                              ")
print("================================================================================")
print("\n")
print("CONFIGURATION:")
print(sprintf("  Method: %s", CONFIG$METHOD))
print(sprintf("  Separate spoken/written: %s", CONFIG$SEPARATE_SPOKEN_WRITTEN))
print(sprintf("  Grammar cutoff: %d", CONFIG$GRAMMAR_CUTOFF))
print(sprintf("  Lexical cutoff: %d", CONFIG$LEXICAL_CUTOFF))
print("")
print("GRAMMAR IMPUTATION:")
print(sprintf("  Participants: %d", nrow(grammar_imputed) - 1))
print(sprintf("  Items: %d", length(grammar_all_items)))
print(sprintf("  Runtime: %.2f minutes", grammar_runtime))
if (!is.na(grammar_oob)) {
  print(sprintf("  OOB error: %.4f", grammar_oob))
}
print("")
print("LEXICAL IMPUTATION:")
print(sprintf("  Participants: %d", nrow(lexical_imputed) - 1))
print(sprintf("  Items: %d", length(lexical_all_items)))
print(sprintf("  Runtime: %.2f minutes", lexical_runtime))
if (!is.na(lexical_oob)) {
  print(sprintf("  OOB error: %.4f", lexical_oob))
}
print("")
print("TOTAL:")
print(sprintf("  Total runtime: %.2f minutes", grammar_runtime + lexical_runtime))
print("")
print("OUTPUT FILES:")
print(paste("  -", basename(output_file)))
print(paste("  -", basename(output_file_lex)))
print("")
if (CONFIG$METHOD == "missForest") {
  print("Method Notes:")
  print("- missForest provides OOB errors for quality assessment")
  print("- Treats grammar data as factors during imputation")
  print("- Uses parallel processing with random forests")
} else {
  print("Method Notes:")
  print("- hdImpute uses hybrid distance-based imputation with PMM")
  print("- Does not provide OOB errors (use validation script for assessment)")
  print("- Handles ordinal data with appropriate distance metrics")
}
print("================================================================================\n")

################################################################################
# UPLOAD TO DATABASE (OPTIONAL)
################################################################################

if (CONFIG$UPLOAD_TO_DB) {
  tryCatch({
    upload_to_database(grammar_imputed, lexical_imputed)
  }, error = function(e) {
    print(paste("ERROR: Database upload failed:", e$message))
    print("Imputation completed successfully, but database upload was not successful")
  })
} else {
  print("\nDatabase upload skipped (UPLOAD_TO_DB = FALSE)")
}

print(paste("\nImputation finished at", Sys.time()))

# Close logging
sink(type = "output")
sink(type = "message")
close(log_conn)
print(paste("Log saved to:", log_file))

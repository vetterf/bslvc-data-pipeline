#!/usr/bin/env Rscript
# BSLVC Export – export SQLite views to RDS files
# Usage:  Rscript BSLVC_export.R <data_directory>

library(data.table)
library(RSQLite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript BSLVC_export.R <data_directory>")
}

data_dir <- args[1]
connString <- file.path(data_dir, "BSLVC_sqlite.db")

if (!file.exists(connString)) {
  stop(paste("Database not found:", connString))
}

dbhandle <- dbConnect(SQLite(), dbname = connString)

Views <- c("Informants", "BSLVC_ALL", "BSLVC_GRAMMAR",
           "BSLVC_SPOKEN", "BSLVC_WRITTEN", "BSLVC_LEXICAL")

for (v in Views) {
  tryCatch({
    res <- as.data.table(dbGetQuery(dbhandle, paste0("SELECT * FROM ", v)))
    out <- file.path(data_dir, paste0(v, ".rds"))
    saveRDS(res, out)
    cat(paste0("  ", v, ".rds  (", nrow(res), " rows)\n"))
  }, error = function(e) {
    cat(paste0("  warning: ", v, ": ", e$message, "\n"))
  })
}

dbDisconnect(dbhandle)
cat("Done\n")

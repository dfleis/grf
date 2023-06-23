package.name <- "grf"

#---------- unload & uninstall existing grf package
if (package.name %in% .packages()) {
  do.call("detach", args = list(name = paste0("package:", package.name), unload = TRUE))
}
if (package.name %in% library()$results[,1]) {
  remove.packages("grf")
}

#---------- build
# Builds and tests the GRF package.
#
# To build the package for development:
#   `Rscript build_package.R`
#
# To prepare a CRAN build:
#   `Rscript build_package.R --as-cran`
start_time <- Sys.time()

args <- commandArgs(TRUE)
library(Rcpp)
library(devtools)
library(testthat)
library(roxygen2)


# If built for CRAN, exlude all test except ones with "cran" in the filename
# by adding the following regex to .Rbuildignore.
if (!is.na(args[1]) && args[1] == "--as-cran") {
  write_union("grf/.Rbuildignore", "^tests/testthat/test_((?!cran).).*")
  write_union("grf/.Rbuildignore", "^tests/testthat/data")
  write_union("grf/.Rbuildignore", "^tests/testthat/Rplots.pdf")
}

# Auto-generate documentation files
roxygen2::roxygenise(package.name)

# Run Rcpp and build the package.
# Symlinks in `grf/src` point to the Rcpp bindings (`grf/bindings`) and core C++ (`core/src`).
# Note: we don't link in third_party/Eigen, because for the R package build we provide
# access to the library through RcppEigen.
compileAttributes(package.name)
clean_dll(package.name)
build(package.name)

# Test installation and run some smoke tests.
install(package.name)
library(package.name, character.only = TRUE)
# # Treat warnings as errors.
# options(warn = 2)
# test_package(package.name)

end_time <- Sys.time()
cat("Start:\t", as.character(start_time), "\nEnd:\t", as.character(end_time), "\n")
beepr::beep(1); Sys.sleep(0.33)
beepr::beep(1); Sys.sleep(0.33)

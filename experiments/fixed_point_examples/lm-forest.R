##############################################################################################################################
# Tests comparing the accuracy of the gradient-based method with the two fixed-point methods for heterogeneous effect
# estimation in the presence of multiple continuous treatment regressors.
#
# NOTES:
#   * Outcome model Y_i = W_i tau(X_i)^T + eps_i for
#       - K-dimensional treatment regressors W_i ~ N(0, 1_K),
#       - p-dimensional auxiliary covariates X_i ~ N(0, 1_p),
#       - eps_i ~ N(0, 1),
#       - tau(x) = (beta_1 x1, beta_2 x1, ..., beta_K x1), so that the effects are linearly heterogeneous in the
#         first feature of the auxiliary covariates x = [x1, ..., xp], and where each beta_k ~ N(0, 1) is fixed across
#         a single replication, but sampled randomly across the replications
#   * Pre-computing residualized quantities Y.hat and W.hat prior to the main lm_forest calls so that we can pass the
#     same result to all three implementations.
#
##############################################################################################################################
#devtools::install_github("dfleis/grf", subdir = "r-package/grf")
library(grf)
library(tidyverse)
library(ggplot2)
library(ggh4x)

###########################################################################################################
################################################## SETUP ##################################################
###########################################################################################################
set.seed(125)
nsims <- 2

cpp.seed    <- 1
num.threads <- 6 # default = NULL, irrelevant if fitting a single tree

num.trees       <- 200
sample.fraction <- 0.5
min.node.size   <- 5 # min node size to be under consideration for a split
honesty         <- T # honest subsampling

methods <- list("grad" = "grad", "fp1" = "fp1", "fp2" = "fp2")

#--------------- grid of data-generating parameters
sig.eps <- 1
n.new   <- 1000 # nb. of test samples

Kv <- c(2, 4, 8) # treatment regressors
pv <- 2              # auxiliary covariates
nv <- c(2500, 5000)  # training samples

pars.grid <- expand.grid("K" = Kv, "p" = pv, "n" = nv)
df.list <- vector(mode = "list", length = nrow(pars.grid))

#################################################################################################################
################################################## SIMULATIONS ##################################################
#################################################################################################################
t0.all <- Sys.time()
for (idx.pars in 1:nrow(pars.grid)) {
  pars <- pars.grid[idx.pars,]

  K <- pars$K
  p <- pars$p
  n <- pars$n

  cat(paste0(Sys.time()), "\tidx.pars =", idx.pars, "of", nrow(pars.grid), "\tK =", K, "\tp =", p, "\tn =", n, "\t")

  t0.sim <- Sys.time()
  sim <- replicate(nsims, {
    #=============== generate data
    beta <- rnorm(K)
    eps  <- rnorm(n, 0, sig.eps)
    X    <- matrix(rnorm(n * p), nrow = n)
    W    <- matrix(rnorm(n * K), nrow = n)

    tauX <- sapply(beta, function(b) b * X[,1])
    Y    <- rowSums(W * tauX) + eps

    # data for test errors
    X.new    <- matrix(rnorm(n.new * p), nrow = n.new)
    tauX.new <- sapply(beta, function(b) b * X.new[,1])

    #=============== fit Y and W forests to estimate Y.hat and W.hat
    forest.Y <- grf::multi_regression_forest(X = X, Y = Y,
                                             num.trees     = max(50, round(num.trees/4)),
                                             min.node.size = 5,
                                             seed          = cpp.seed)
    forest.W <- grf::multi_regression_forest(X = X, Y = W,
                                             num.trees     = max(50, round(num.trees/4)),
                                             min.node.size = 5,
                                             seed          = cpp.seed)
    Y.hat <- predict(forest.Y)$predictions
    W.hat <- predict(forest.W)$predictions

    #=============== fit main forests
    forests <- lapply(sample(methods), function(m) { # randomize method order
      pt <- proc.time()
      forest.fit <- lm_forest(X = X, Y = Y, W = W, Y.hat = Y.hat, W.hat = W.hat,
                              num.trees       = num.trees,
                              sample.fraction = sample.fraction,
                              min.node.size   = min.node.size,
                              honesty         = honesty,
                              ci.group.size   = 1,
                              compute.oob.predictions = F,
                              method          = m,
                              num.threads     = num.threads,
                              seed            = cpp.seed)
      time.fit      <- (proc.time() - pt)["elapsed"]
      avg.nb.splits <- mean(sapply(forest.fit$`_split_vars`, length))

      #===== oob errors
      tauX.hat <- predict(forest.fit)$predictions[,,1]
      oob.err  <- mean(sqrt(rowMeans((tauX.hat - tauX)^2)))

      #===== test errors
      tauX.new.hat <- predict(forest.fit, newdata = X.new)$predictions[,,1]
      test.err     <- mean(sqrt(rowMeans((tauX.new.hat - tauX.new)^2)))

      return (list("time" = unname(time.fit), "avg.nb.splits" = avg.nb.splits, "oob.err" = oob.err, "test.err" = test.err))
    })
    forests <- forests[order(match(names(forests), names(methods)))] # re-order according to the original vector
    return (forests)
  })
  tm.sim <- Sys.time() - t0.sim
  print(tm.sim)

  #=============== extract & restructure simulation outputs
  df.list[[idx.pars]] <- apply(sim, 1, bind_rows) %>%
    map2(.x = ., .y = names(.), ~ mutate(.x, method = .y)) %>% # add column in each df with the corresponding method name
    bind_rows() %>%
    mutate(K = K, p = p, n = n)
}
t1.all <- Sys.time()
(tm.all <- t1.all - t0.all)

df <- bind_rows(df.list) %>%
  mutate(method2 = factor(method, levels = methods))

##################################################################################################################
################################################### DRAW PLOTS ###################################################
##################################################################################################################

ggplot(df, aes(x = as.factor(K), y = test.err, color = method2, fill = method2)) +
  geom_boxplot() +
  scale_color_manual(values = pals::brewer.dark2(3)) +
  scale_fill_manual(values = pals::brewer.pastel2(3)) +
  ggh4x::facet_nested("Auxiliary Covariate Features (p)" + p ~ "Training Samples (n)" + n)
















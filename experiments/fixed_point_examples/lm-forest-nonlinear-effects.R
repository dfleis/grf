########################################################################################################################
# Tests comparing the gradient and fixed-point methods' ability to estimate highly curved treatment effects.
#   * lm_forest
#
# NOTES:
#   *
#   *
#
########################################################################################################################
library(grf)
library(tidyverse)
library(ggplot2)
library(ggh4x)

###########################################################################################################
################################################## SETUP ##################################################
###########################################################################################################
set.seed(124)
nsims <- 50
cpp.seed    <- 1
num.threads <- 6 # default = NULL, irrelevant if fitting a single tree

num.trees       <- 2000
sample.fraction <- 0.5
min.node.size   <- 5 # min node size to be under consideration for a split
honesty         <- T # honest subsampling

methods <- list("grad" = "grad", "fp1" = "fp1", "fp2" = "fp2")

#--------------- data-generating parameters
n.new <- 1000
p <- 2
nv <- c(2500, 5000)
sv <- 10^seq(-4, 0, length.out = 5) # treatment effect concentrations about 0.5 (std dev of normal pdf)
Kv <- c(2, 4, 8, 16)

tau_k <- function(x, sigma) exp(-0.5 * (x[1] - 0.5)^2/sigma^2) # normal pdf(mu, sigma) rescaled to maximum 1 for all sigma
tau   <- function(x, sigma, n.levels) {
  if (is.vector(x)) rep(tau_k(x, sigma), times = n.levels)
  else t(apply(x, 1, tau, sigma = sigma, n.levels = n.levels))
}


pars.grid <- expand.grid("s" = sv, "K" = Kv, "n" = nv)
df.list <- vector(mode = "list", length = nrow(pars.grid))

#################################################################################################################
################################################## SIMULATIONS ##################################################
#################################################################################################################
t0.all <- Sys.time()
for (idx.pars in 1:nrow(pars.grid)) {
  pars <- pars.grid[idx.pars,]

  s <- pars$s
  K <- pars$K
  n <- pars$n

  cat(paste0(Sys.time()), "\tidx.pars =", idx.pars, "of", nrow(pars.grid), "\ts =", s, "\tK =", K, "\tn =", n, "\t")

  t0.sim <- Sys.time()
  sim <- replicate(nsims, {
    ### training data
    X <- matrix(runif(n * p), nrow = n)
    W <- matrix(runif(n * K), nrow = n)
    tauX <- tau(X, sigma = s, n.levels = K)
    eps  <- rnorm(n)
    Y <- rowSums(W * tauX) + eps

    ### testing data
    X.new    <- matrix(runif(n.new * p), nrow = n.new)
    tauX.new <- tau(X.new, sigma = s, n.levels = K)


    #--------------- fit conditional mean forests of Y|X and W|X
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

    #--------------- fit main forests
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

  #--------------- extract & restructure simulation outputs
  df.list[[idx.pars]] <- apply(sim, 1, bind_rows) %>%
    map2(.x = ., .y = names(.), ~ mutate(.x, method = .y)) %>% # add column in each df with the corresponding method name
    bind_rows() %>%
    mutate(s = s, K = K, p = p, n = n)
}
t1.all <- Sys.time()
(tm.all <- t1.all - t0.all)

df <- bind_rows(df.list) %>%
  mutate(method2 = factor(method, levels = methods))


##################################################################################################################
################################################### DRAW PLOTS ###################################################
##################################################################################################################

ggplot(df, aes(x = factor(log10(s)), y = test.err, color = method2, fill = method2)) +
  geom_boxplot() +
  scale_color_manual(values = pals::brewer.dark2(3)) +
  scale_fill_manual(values = pals::brewer.pastel2(3)) +
  ggh4x::facet_nested("Treatment Regressors (K)" + K ~ "Training Samples (n)" + n)



ggplot(df, aes(x = factor(log10(s)), y = test.err, color = method2, fill = method2)) +
  geom_boxplot() +
  scale_color_manual(values = pals::brewer.dark2(3)) +
  scale_fill_manual(values = pals::brewer.pastel2(3)) +
  facet_grid(rows = vars(K), cols = vars(method2))

ggplot(df, aes(x = method2, y = test.err, color = method2, fill = method2)) +
  geom_boxplot() +
  scale_color_manual(values = pals::brewer.dark2(3)) +
  scale_fill_manual(values = pals::brewer.pastel2(3)) +
  facet_grid(rows = vars(K), cols = vars(factor(log10(s) )))






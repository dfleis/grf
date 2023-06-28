##############################################################################################################################
# Tests comparing the speed of the gradient-based method with the two fixed-point methods for heterogeneous effect
# estimation in the presence of multiple continuous treatment regressors. The primary goal is to compare the fit times of
# the "gradient tree" with the two implementations of the "fixed-point tree" algorithm, and so we begin by considering
# forests of B = 1 tree with no subsampling.
#
# NOTES:
#   * Outcome model Y_i = W_i tau(X_i)^T + eps_i for
#       - K-dimensional treatment regressors W_i ~ N(0, 1_K),
#       - p-dimensional auxiliary covariates X_i ~ N(0, 1_p),
#       - eps_i ~ N(0, 1),
#       - tau(x) = (beta_1 x1, beta_2 x1, ..., beta_K x1), so that the effects are linearly heterogeneous in the
#         first feature of the auxiliary covariates x = [x1, ..., xp], and where each beta_k ~ N(0, 1) is fixed across
#         a single replication, but sampled randomly across the replications
#   * Pre-computing residualizing quantities Y.hat and W.hat outside of the main lm_forest call, setting them both to 0 for
#     the sake of time (since this test is looking exclusively at speed and not accuracy).
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
set.seed(124)
nsims <- 100

cpp.seed    <- 1
num.threads <- 1

sample.fraction <- 1 # default = 0.5, training samples used to fit each tree (drawn without replacement)
min.node.size   <- 5 # default = 5, min size of a node for it to be under consideration for splitting
honesty         <- F # default = T, honest subsampling

#methods <- factor(c("grad", "fp1", "fp2"), levels = c("grad", "fp1", "fp2"))
methods <- list("grad" = "grad", "fp1" = "fp1", "fp2" = "fp2")

#--------------- grid of data-generating parameters
sig.eps <- 1

Kv <- c(2, 4, 8, 16) # treatment regressors
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

  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      "\tidx.pars =", idx.pars, "of", nrow(pars.grid), "\tK =", K, "\tp =", p, "\tn =", n, "\t")

  Y.hat <- matrix(0, nrow = n, ncol = 1)
  W.hat <- matrix(0, nrow = n, ncol = K)

  t0.sim <- Sys.time()
  sim <- replicate(nsims, {
    #===============  generate data
    beta <- rnorm(K)
    eps  <- rnorm(n, 0, sig.eps)
    X    <- matrix(rnorm(n * p), nrow = n)
    W    <- matrix(rnorm(n * K), nrow = n)

    tauX <- sapply(beta, function(b) b * X[,1])
    Y    <- rowSums(W * tauX) + eps

    #=============== fit main trees
    trees <- lapply(sample(methods), function(m) { # randomize method order
      pt <- proc.time()
      tree.fit <- lm_forest(X = X, Y = Y, W = W, Y.hat = Y.hat, W.hat = W.hat,
                            num.trees       = 1,
                            sample.fraction = sample.fraction,
                            min.node.size   = min.node.size,
                            honesty         = honesty,
                            ci.group.size   = 1,
                            compute.oob.predictions = F,
                            method          = m,
                            num.threads     = num.threads,
                            seed            = cpp.seed)
      time.fit  <- (proc.time() - pt)["elapsed"]
      nb.splits <- sapply(tree.fit$`_split_vars`, length)
      return (list("time" = unname(time.fit), "nb.splits" = nb.splits))
    })
    trees <- trees[order(match(names(trees), names(methods)))] # re-order according to the original vector
    return (trees)
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
df.time.ratio <- df %>%
  select(-c("nb.splits")) %>%
  group_by(K, p, n) %>%
  reframe("fp1/grad" = time[method == "fp1"]/time[method == "grad"],
          "fp2/grad" = time[method == "fp2"]/time[method == "grad"]) %>%
  gather(method, time.ratio, `fp1/grad`:`fp2/grad`) %>%
  group_by(K, p, n, method) %>%
  mutate(hline = 1)

ggplot(df.time.ratio, aes(x = as.factor(K), y = time.ratio, color = method, fill = method)) +
  ggtitle(paste0("lm_forest: Fixed-Point vs. Gradient-Based Tree Fit Times"),
          subtitle = paste0("One tree, ", nsims, " replications")) +
  scale_y_continuous(limits = c(0, 1.25)) +
  geom_hline(aes(yintercept = hline), col = 'gray50', linewidth = 0.85) +
  geom_boxplot(outlier.alpha = 0) +
  scale_color_manual(values = pals::brewer.dark2(3)[-1]) +
  scale_fill_manual(values = pals::brewer.pastel2(3)[-1]) +
  theme(legend.position = c(0.0875, 0.2)) +
  labs(fill = "Comparison", color = "Comparison") +
  ylab("Relative Fit Time") +
  theme(legend.text = element_text(family = "monospace")) +
  ggh4x::facet_nested("Auxiliary Covariate Features (p)" + p ~ "Training Samples (n)" + n)








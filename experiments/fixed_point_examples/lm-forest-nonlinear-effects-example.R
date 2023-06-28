########################################################################################################################
# Single example illustrating the fits provided by lm_forest using the gradient and fixed-point implementations.
#   * Multivariate continuous treatment model (fit via lm_forest)
#   * Highly curved treatment effects.
#
# NOTES:
#   *
#   *
#
########################################################################################################################
library(grf)

###########################################################################################################
################################################## SETUP ##################################################
###########################################################################################################
set.seed(124)
#--------------- FOREST PARAMETERS
cpp.seed    <- 1
num.threads <- NULL # default = NULL, irrelevant if fitting a single tree

num.trees       <- 2000
sample.fraction <- 0.5
min.node.size   <- 5 # min node size to be under consideration for a split
honesty         <- T # honest subsampling

methods <- list("grad" = "grad", "fp1" = "fp1", "fp2" = "fp2")

#--------------- DATA-GENERATING PARAMETERS
n <- 2500 # training samples
p <- 1    # auxiliary covariate features
K <- 2    # (continuous) treatment regressors
sig.eps    <- 0.0
nb.periods <- 1 # periodicity of the treatment effects tau_k(x) = sin(2 * pi * nb.periods * x[1]), controlling some notion of nonlinearity/curvature
n.new      <- 1000 # number of test samples for computing test error estimates (randomly sampled)
n.new.seq  <- 100   # number of test samples for plotting test fits (equally spaced in x1, any other covariates set to 0)

tau_k   <- function(x, nb.periods) sin(2 * pi * nb.periods * x[1]) # k-th treatment effect (heterogeneous in x1)
tau_vec <- function(x, nb.vars, ...) rep(tau_k(x = x, ...), times = nb.vars) # vector of K treatment effects (all identical)
tau     <- function(x, nb.vars, ...) {
  if (is.vector(x)) tau_vec(x = x, nb.vars = nb.vars, ...)
  else t(apply(x, 1, tau, nb.vars = nb.vars, ...))
}

#--------------- GENERATE DATA
##### training data
X <- matrix(runif(n * p), nrow = n)
W <- matrix(runif(n * K), nrow = n)
tauX <- tau(X, nb.vars = K, nb.periods = nb.periods)
eps  <- sig.eps * rnorm(n)
Y <- rowSums(W * tauX) + eps

##### testing data
X.new     <- matrix(runif(n.new * p), nrow = n.new)
X.new.seq <- matrix(0, nrow = n.new.seq, ncol = p)
X.new.seq[,1] <- seq(0, 1, length.out = n.new.seq)
tauX.new      <- tau(X.new, nb.vars = K, nb.periods = nb.periods)
tauX.new.seq  <- tau(X.new.seq, nb.vars = K, nb.periods = nb.periods)


#############################################################################################################
################################################ FIT FORESTS ################################################
#############################################################################################################
#--------------- CONDITIONAL MEAN FORESTS Y|X, W|X
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

#--------------- MAIN FORESTS
pt <- proc.time()
forests <- lapply(methods, function(m) {
  # fit forest
  forest <- lm_forest(X = X, Y = Y, W = W, Y.hat = Y.hat, W.hat = W.hat,
                      num.trees       = num.trees,
                      sample.fraction = sample.fraction,
                      min.node.size   = min.node.size,
                      honesty         = honesty,
                      ci.group.size   = 1,
                      compute.oob.predictions = F,
                      method          = m,
                      num.threads     = num.threads,
                      seed            = cpp.seed)
})
(tm <- proc.time() - pt)

#--------------- COMPUTE PREDICTIONS
tauX.new.hat     <- lapply(forests, function(forest) predict(forest, newdata = X.new)$prediction[,,1])
tauX.new.seq.hat <- lapply(forests, function(forest) predict(forest, newdata = X.new.seq)$prediction[,,1])


####################################################################################################################
################################################### DRAW FIGURES ###################################################
####################################################################################################################
### setup plot aesthetics
k.plot <- 1 # fitted effects should all be more or less identical to one another, k = 1, ..., K
point.cex <- 0.75
point.labels <- names(forests)
point.colors <- rainbow(3)#c("firebrick3", "limegreen", "orchid4")
point.shapes <- c(22, 21, 24)

x <- seq(0, 1, length.out = 1e3)
tauX.true <- sapply(x, tau_k, nb.periods = nb.periods)

xlims <- c(0, 1)
ylims <- range(-1, 1, tauX.new.seq.hat, tauX.true)

plot(NA, xlim = xlims, ylim = ylims, xaxt = "n", yaxt = "n", xlab = "", ylab = "")
grid(); abline(h = 0, v = 0, lwd = 1.5, col = "gray50")
axis(side = 1, line = -0.5, tick = F); axis(side = 1, labels = NA)
axis(side = 2, line = -0.5, tick = F); axis(side = 2, labels = NA)
mtext(expression(X[1]), side = 1, at = 0.5, line = 1.75)
mtext("Treatment Effect", side = 2, at = mean(ylims), line = 1.75)

lines(tauX.true ~ x, lwd = 1.5)
for (m in 1:length(tauX.new.seq.hat)) {
  points(tauX.new.seq.hat[[m]][,k.plot] ~ X.new.seq[,1], pch = point.shapes[m], col = point.colors[m], lwd = 2, cex = point.cex)
}
legend("topright", title = "Method", legend = methods, col = point.colors, pch = point.shapes,
       lwd = 2, lty = NA, seg.len = 1, cex = point.cex)



# tauX.new.seq.hat$grad - tauX.new.seq.hat$fp1
# tauX.new.seq.hat$fp1 - tauX.new.seq.hat$fp2


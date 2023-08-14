#' Local linear forest tuning
#'
#' Finds the optimal ridge penalty for local linear prediction.
#'
#' @param forest The forest used for prediction.
#' @param linear.correction.variables Variables to use for local linear prediction. If left null,
#'          all variables are used. Default is NULL.
#' @param ll.elnet.alpha The elastic net mixing parameter, a proportion bound within 0 and 1. The parameter \code{ll.elnet.alpha}
#'                       is defined as \eqn{\alpha} in the elastic net penalty parameterization
#'                       \deqn{(1-\alpha)/2||\beta||_2^2+\alpha||\beta||_1.} \code{alpha=1} is the lasso penalty,
#'                       and \code{alpha=0} the ridge penalty.
#' @param ll.weight.penalty Option to standardize ridge penalty by covariance (TRUE),
#'                          or penalize all covariates equally (FALSE). Defaults to FALSE.
#' @param num.threads Number of threads used in training. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param lambda.path Optional list of lambdas to use for cross-validation.
#' @param thresh Convergence threshold for coordinate descent. Each coordinate descent loop continues until the
#'               maximum change in the objective after any coefficient update is less than \code{thresh} times
#'               the null deviance.
#' @param maxit Maximum number of passes over the data for all lambda values.
#' @return A list of lambdas tried, corresponding errors, and optimal ridge penalty lambda.
#'
#' @keywords internal
tune_ll_regression_forest2 <- function(forest,
                                       linear.correction.variables = NULL,
                                       ll.elnet.alpha = NULL,
                                       ll.weight.penalty = FALSE,
                                       num.threads = NULL,
                                       lambda.path = NULL,
                                       thresh = 1e-07,
                                       maxit = 1e5,
                                       ...) {
  forest.short <- forest[-which(names(forest) == "X.orig")]
  X <- forest[["X.orig"]]
  Y <- forest[["Y.orig"]]
  train.data <- create_train_matrices(X, outcome = Y)

  # Validate variables
  num.threads <- validate_num_threads(num.threads)
  linear.correction.variables <- validate_ll_vars(linear.correction.variables, ncol(X))
  ll.elnet.alpha <- validate_ll_elnet_alpha(ll.elnet.alpha)
  thresh <- validate_ll_thresh(thresh)
  maxit <- validate_ll_maxit(maxit)
  ll.lambda <- validate_ll_path(lambda.path)

  # Subtract 1 to account for C++ indexing
  linear.correction.variables <- linear.correction.variables - 1

  args <- list(forest.object = forest.short,
               num.threads = num.threads,
               estimate.variance = FALSE,
               ll.elnet.alpha = ll.elnet.alpha,
               ll.lambda = ll.lambda,
               ll.weight.penalty = ll.weight.penalty,
               linear.correction.variables = linear.correction.variables,
               thresh = thresh,
               maxit = maxit)

  prediction.object <- do.call.rcpp(ll_regression_predict_oob2, c(train.data, args))
  predictions <- prediction.object$predictions
  errors <- apply(predictions, MARGIN = 2, FUN = function(row) {
    mean((row - Y)**2)
  })

  return(list(
    lambdas = ll.lambda, errors = errors, oob.predictions = predictions,
    lambda.min = ll.lambda[which.min(errors)]
  ))
}

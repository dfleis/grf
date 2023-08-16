/*-------------------------------------------------------------------------------
 * Wrapper for the glmnetpp core tools for elastic net base on the elnet_exp
 * C++ binding found in the glmnet package
 * https://github.com/cran/glmnet/blob/e85bab25e05c0d33095d71dcd114328ca25128eb/src/elnet_exp.cpp
 *
 * Guide to glmnet's variable/argument naming scheme:
 *
 * Inside glmnet::glmnet (R code)
 *    dfmax: Max number of variables to be included in the model.
 *    pmax: Max number of variables permitted to be nonzero.
 *    exclude: Indices of variables to explicitly exclude from the model.
 *    penalty.factor: Differential penalization/shrinkage factors applied to each
 *                    coefficient. Value of 0 means no penalization, value of Inf
 *                    is equivalent to adding the corresponding index to 'exclude'
 *    lower.limits: Lower limits for each coefficient (default -Inf). Can be a
 *                  vector of size nvars or a scalar.
 *    upper.limits: Upper limits, analogous to lower.limits, defaulting to +Inf.
 *    type.gaussian: Algorithm type for gaussian models, either "covariance" or
 *                   "naive". Defaults to "covariance" when nvar < 500.
 *
 * Inside glmnet's primary C++ binding elnet_exp
 * https://github.com/cran/glmnet/blob/e85bab25e05c0d33095d71dcd114328ca25128eb/src/elnet_exp.cpp
 *    int ka -- Flag mapping to type.gaussian (1 = "covariance", 2 = "naive")
 *    double parm -- rename of alpha
 *    Eigen::MatrixXd x -- Predictors, not yet (optionally) centered/standardized (done in the C++ code, not R).
 *    Eigen::VectorXd y -- Response, not yet centered/standardized (done in the C++ code, not R).
 *    Eigen::VectorXd w -- Weights, not yet rescaled to sum to nobs.
 *    const Eigen::Map<Eigen::VectorXi> jd -- Vector of length num_excluded_vars + 1. The first entry
 *                                            contains the value of num_excluded_vars and the remaining
 *                                            entries contain the indices of excluded variables as well
 *                                            as the indices of variables given an infinite penalty.factor.
 *    const Eigen::Map<Eigen::VectorXd> vp -- as.double(penalty.factor)
 *    int ne -- rename of dfmax
 *    int nx -- rename of pmax
 *    int nlam -- Number of lambdas in either the user-specified sequence, or the target number of lambdas
 *                for glmnet to create (might stop early).
 *    double flmin -- Validated version of lambda.min.ratio. If lambda is specified, then lambda.min.ratio is
 *                    ignored and flmin = 1. Otherwise, flmin = as.double(lambda.min.ratio), unless
 *                    lambda.min.ratio >= 1, in which case glmnet::glmnet stops.
 *    const Eigen::Map<Eigen::VectorXd> ulam -- User specified lambda(s).
 *    double thr -- rename of thresh
 *    int isd -- Flag for whether the predictors should be rescaled (standardize = T), 0 = no, 1 = yes.
 *    int intr -- Flag for whether the predictors should be centered (intercept = T), 0 = no, 1 = yes.
 *    int maxit -- maxit
 *    SEXP pb -- Object to enable glmnet's progress bar via the trace.it argument.
 *    int lmu -- Tracks the size of the lambda path/sequence fit by glmnet (glmnet can stop before the target
 *               length of nlam if the change in deviance from the previous lambda is sufficiently small).
 *    Eigen::Map<Eigen::VectorXd> a0 -- Size-[nlam] vector to store the estimated intercepts.
 *    Eigen::Map<Eigen::MatrixXd> ca -- Size-[nx-by-nlam] container to store the estimated slopes.
 *    Eigen::Map<Eigen::VectorXi> ia -- Size-[nx] vector to store the ever-active indices (ever-active set) at a given lambda.
 *    Eigen::Map<Eigen::VectorXi> nin -- Size-[nlam] vector to track the size of the active set (nb. coefs
 *                                       that have ever been nonzero at previous lambdas).
 *    Eigen::Map<Eigen::VectorXd> rsq -- Size-[nlam] vector to track model deviance.
 *    Eigen::Map<Eigen::VectorXd> alm -- Size-[nlam] vector to track internally-created lambda sequence.
 *    int nlp -- Total number of coordinate descent passes over the data, summed over all lambda values.
 *    int jerr -- Error flag.
 #-------------------------------------------------------------------------------*/
#include <cstddef> // [[ TODO ]] check if these libraries are all still required
#include <cmath>
#include <string>
 
#include <RcppEigen.h>
#include <R.h>
#include <Rinternals.h>
 
#include "glmnetpp_bits/glmnetpp"
#include "driver.h"
#include "internal.h"
 
// TESTING
#include <unistd.h>
#include <iostream>

using namespace Rcpp;

inline void dummy_setpb(int) {} // disable the (optional) glmnet progress bar

inline void check_jerr(int n, int maxit, int pmax) {
  // https://github.com/cran/glmnet/blob/e85bab25e05c0d33095d71dcd114328ca25128eb/R/jerr.R
  // glmnet's warning system is done in R after having called the primary C++ binding
  // here, we warn inside C++ since the elastic net model is part of a larger procedure
  std::string msg = "from glmnetpp C++ code (error code " + std::to_string(n) + "); ";
  if (n > 0) { // fatal error
    if (n < 7777) msg += "Memory allocation error; contact package maintainer.";
    else if (n == 7777) msg += "All used predictions have zero variance";
    else if (n == 10000) msg += "All penalty factors are <= 0.";
    else msg += "Unknown error.";
    Rcpp::stop((msg + "\n").c_str());
  } else if (n < 0) { // non-fatal error
    if (n > -10000) {
      msg += "Convergence for " + std::to_string(-n) +
        "th lambda value not reached after maxit = " + std::to_string(maxit) +
        " iterations; solutions for larger lambdas returned.";
    }
    if (n < -10000) {
      msg += "Number of nonzero coefficients along the path exceeds pmax = " +
        std::to_string(pmax) + " at " + std::to_string(-n-10000) +
        "th lambda value; solutions " +
        "for larger lambdas returned";
    }
    REprintf((msg + "\n").c_str());
  }
}

namespace grf {

struct ElnetWrapper {
private:
  static const bool RESCALE_PREDICTORS = false;
  static const bool CENTER_PREDICTORS = true;

public:
  static void fit(std::vector<double>& preds,
                  Eigen::MatrixXd x,
                  Eigen::VectorXd y,
                  Eigen::VectorXd w,
                  double alpha,
                  int nlam,
                  std::vector<double> lambdas,
                  double thresh,
                  bool weight_penalty,
                  int maxit) {
    if (w.size() == 1) { // glmnet doesn't handle the case of a single observation
      for (int i = 0; i < preds.size(); i++) { 
        preds[i] = y(0);
      }
      return;
    }

    int nobs = x.rows();
    int nvars = x.cols();

    // [[ TODO ]] default glmnet arguments, presumably I can make these user-specifiable
    double lambda_min_ratio = nobs < nvars ? 1e-02 : 1e-04; // only relevant when nlam > 1
    int dfmax = nvars + 1;
    int pmax = std::min(dfmax * 2 + 20, nvars);
    int type_gaussian = nvars < 500 ? 1 : 2; // algorithm type (see arg type.gaussian of glmnet::glmnet)
    double lower_limits = -::InternalParams().big;
    double upper_limits = ::InternalParams().big;

    // glmnet allows users to explicitly exclude features from the model via the argument 'exclude'
    // grf already takes care of the excluding, but I've kept num_excluded_vars for readability
    int num_excluded_vars = 0;

    // rename variables to match the argument names in grf:::elnet_exp
    Eigen::VectorXi jd = Eigen::VectorXi::Zero(1);
    Eigen::Vector2d cl_col(lower_limits, upper_limits);
    Eigen::MatrixXd cl(2, nvars);
    cl.colwise() = Eigen::Map<Eigen::VectorXd>(cl_col.data(), cl_col.size());
    int ne = dfmax;
    int nx = pmax;
    // double flmin = nlam > 1 ? lambda_min_ratio : 1.0;
    double flmin = 1.0; // when lambda is specified, set flmin to 1.0

    // initialize & format empty data structures for the glmnetpp elastic net driver
    // [[ TODO ]] might make more sense to do this outside of the wrapper to avoid re-allocations
    // [[ TODO ]] or possibly a set of static variables/constants defined outside of this function
    Eigen::Map<Eigen::VectorXd> ulam(lambdas.data(), lambdas.size());
    Eigen::Map<Eigen::VectorXd> a0(preds.data(), preds.size()); // convert
    int lmu = 0;
    Eigen::MatrixXd ca = Eigen::MatrixXd::Zero(nx, nlam);
    Eigen::VectorXi ia = Eigen::VectorXi::Zero(nx);
    Eigen::VectorXi nin = Eigen::VectorXi::Zero(nlam);
    Eigen::VectorXd rsq = Eigen::VectorXd::Zero(nlam);
    Eigen::VectorXd alm = Eigen::VectorXd::Zero(nlam);
    int nlp = 0;
    int jerr = 0;

    // prior to entering the C++ source, glmnet ensures that the null deviance > 0
    // https://github.com/cran/glmnet/blob/e85bab25e05c0d33095d71dcd114328ca25128eb/R/elnet.R
    double ybar = (y.array() * w.array()).sum() / w.size();
    double sdy = std::sqrt( (((y.array() - ybar).abs2() * w.array()).sum()) / w.size());
    if (sdy == 0) Rcpp::stop("y is constant within a leaf; Gaussian glmnet fails at the standardization step.\n"); 
    
    // compute normalization & penalty factors to be consistent with the existing grf implementation (corresponding to ridge/alpha = 0)
    // We only compute the M matrix to be consistent with the existing LLR method (with alpha = 0 for ridge). Surely, we can do some other normalization
    // that won't require us to compute an otherwise unnecessary product? Note that the X used below is the local X - x0, but not the fully mean-centered matrix.
    Eigen::MatrixXd M = x.transpose() * w.asDiagonal() * x / w.size();
    double normalization = M.trace();
    Eigen::VectorXd penalty_factor = Eigen::VectorXd::Ones(nvars);
    if (!weight_penalty) {
      normalization = (1.0 + normalization) / (1.0 + nvars);
    } else {
      normalization = normalization / nvars;
      penalty_factor = M.diagonal().array() / normalization;
    }
    ulam = ulam.array() * sdy * normalization; // divide lambda by sd(Y) to rescale glmnet ridge outputs to correspond to grf's ridge model
    
    // setup and call the glmnetpp elastic net solver for gaussian models and nonsparse predictor matrix
    using elnet_driver_t = glmnetpp::ElnetDriver<glmnetpp::util::glm_type::gaussian>;
    elnet_driver_t driver;
    auto f = [&]() {
      driver.fit(type_gaussian == 2,
                 alpha,
                 x,
                 y,
                 w,
                 jd,
                 penalty_factor, // [[ TODO ]] user-specified penalty factor beyond the naive & covariance penalty list in grf? is this the same thing as glmnet's 'type.gaussian' argument?
                 cl,
                 ne,
                 nx,
                 nlam,
                 flmin,
                 ulam,
                 thresh,
                 RESCALE_PREDICTORS,
                 CENTER_PREDICTORS,
                 maxit,
                 lmu,
                 a0,
                 ca,
                 ia,
                 nin,
                 rsq,
                 alm,
                 nlp,
                 jerr,
                 dummy_setpb,
                 ::InternalParams());
    };
    run(f, jerr);
    check_jerr(jerr, maxit, pmax);
    if (lmu < 1) REprintf("An empty model has been returned; probably a convergence issue.\n");
  }

};


} // namespace grf

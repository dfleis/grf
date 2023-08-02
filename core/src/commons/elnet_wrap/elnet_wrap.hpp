/*-------------------------------------------------------------------------------
 * Wrapper for the glmnetpp core tools for elastic net base on the elnet_exp
 * C++ binding found in the glmnet package
 * https://github.com/cran/glmnet/blob/e85bab25e05c0d33095d71dcd114328ca25128eb/src/elnet_exp.cpp
 *
 #-------------------------------------------------------------------------------*/
// [[ TODO ]] check if these libraries are all still required
#include <cstddef>
#include <cmath>
#include <RcppEigen.h>
#include "glmnetpp_bits/glmnetpp"
#include <R.h>
#include <Rinternals.h>
#include "driver.h"
#include "internal.h"

using namespace Rcpp;

inline void dummy_setpb(int) {} // disable the (optional) glmnet progress bar

namespace grf {

struct ElnetWrap {
// private:
//   static const bool RESCALE_PREDICTORS = false;
//   static const bool CENTER_PREDICTORS = true;

public:
  static void wrap(std::vector<double>& preds,
                   double alpha,
                   Eigen::MatrixXd x,
                   Eigen::VectorXd y,
                   Eigen::VectorXd w,
                   int nlam,
                   std::vector<double> lambdas, // might need to map to a Eigen::VectorXd at one point below
                   bool weight_penalty,
                   double thresh,
                   int maxit) {
    if (w.size() == 1) { // glmnet doesn't handle the trivial case of a single observation
      for (int i = 0; i < preds.size(); i++) {
        preds[i] = y(0);
      }
      return;
    }

    // format data structures for glmnetpp
    Eigen::Map<Eigen::VectorXd> ulam(lambdas.data(), lambdas.size());
    Eigen::Map<Eigen::VectorXd> a0(preds.data(), preds.size()); // Eigen::VectorXd::Map(&preds[0], a0.size()) = a0;

    // [[ TODO ]] placeholder data and data structures
    int nobs = x.rows();
    int nvars = x.cols();
    int num_excluded_vars = 0;
    double big = 9.9e35; //auto big = ::InternalParams().big;
    int dfmax = nvars + 1;
    int pmax = std::min(dfmax * 2 + 20, nvars);
    double lambda_min_ratio = nobs < nvars ? 1e-02 : 1e-04;

    int ka = 2;
    // [[ TODO ]] are these supposed to be maps?
    Eigen::VectorXi jd = Eigen::VectorXi::Zero(num_excluded_vars + 1); // [[ TODO ]] potential problem for the arguments that are expected to be either CONST or Eigen::Map (or both)
    Eigen::Vector2d cl_col(-big, big);
    Eigen::MatrixXd cl(2, nvars);
    cl.colwise() = Eigen::Map<Eigen::VectorXd>(cl_col.data(), cl_col.size());
    int ne = dfmax;
    int nx = pmax;
    double flmin = nlam > 1 ? lambda_min_ratio : 1;
    bool RESCALE_PREDICTORS = false;
    bool CENTER_PREDICTORS = true;
    int lmu = 0;
    // [[ TODO ]] initialize these things to something? are the dimensions correct?
    // are these supposed to be maps?
    Eigen::MatrixXd ca = Eigen::MatrixXd::Zero(nobs, nlam);
    Eigen::VectorXi ia = Eigen::VectorXi::Zero(nx);
    Eigen::VectorXi nin = Eigen::VectorXi::Zero(nlam);
    Eigen::VectorXd rsq = Eigen::VectorXd::Zero(nlam);
    Eigen::VectorXd alm = Eigen::VectorXd::Zero(nlam);
    int nlp = 0;
    int jerr = 0;

    // compute normalization & penalty factors to be consistent with the existing grf implementation (corresponding to ridge/alpha = 0)
    // We only compute the M matrix to be consistent with the existing LLR method (with alpha = 0 for ridge). Surely, we can do some other normalization
    // that won't require us to compute an otherwise unnecessary product? Note that the X used below is the local X - x0, but not the fully mean-centered matrix.
    double ybar = (y.array() * w.array()).sum() / w.size();
    double sdy = std::sqrt( (((y.array() - ybar).abs2() * w.array()).sum()) / w.size());
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

    // setup and call the glmnetpp elastic net solver for gaussian models and nonsparse x
    using elnet_driver_t = glmnetpp::ElnetDriver<glmnetpp::util::glm_type::gaussian>;
    elnet_driver_t driver;
    driver.fit(ka == 2,
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
  }

};


} // namespace grf

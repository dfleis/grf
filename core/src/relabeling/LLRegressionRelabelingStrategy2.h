/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_LLREGRESSIONRELABELINGSTRATEGY2_H
#define GRF_LLREGRESSIONRELABELINGSTRATEGY2_H

#include "relabeling/RelabelingStrategy.h"

namespace grf {

class LLRegressionRelabelingStrategy2 final: public RelabelingStrategy {
public:
  LLRegressionRelabelingStrategy2(double alpha,
                                  double split_lambda,
                                  bool weight_penalty,
                                  const std::vector<double>& overall_beta,
                                  size_t ll_split_cutoff,
                                  std::vector<size_t> ll_split_variables,
                                  double thresh,
                                  int maxit);
  bool relabel(
      const std::vector<size_t>& samples,
      const Data& data,
      Eigen::ArrayXXd& responses_by_sample) const;
private:
    double alpha;
    double split_lambda;
    bool weight_penalty;
    const std::vector<double>& overall_beta;
    size_t ll_split_cutoff;
    std::vector<size_t> ll_split_variables;
    double thresh;
    int maxit;
};

} // namespace grf

#endif // GRF_LLREGRESSIONRELABELINGSTRATEGY2_H

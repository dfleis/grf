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

#ifndef GRF_MULTICAUSALRELABELINGSTRATEGYFP1_H
#define GRF_MULTICAUSALRELABELINGSTRATEGYFP1_H

#include <vector>

#include "commons/Data.h"
#include "relabeling/RelabelingStrategy.h"
#include "tree/Tree.h"

namespace grf {

/**
 * TO DO... some documentation in the same style as the description found in the original
 * gradient-based MultiCausalRelabelingStrategy.h
 * Exact fixed-point pseudo-outcome calculation for heterogeneous treatment effect
 * estimation. See algorithm 'fp1' in https://arxiv.org/abs/2306.11908
 */
class MultiCausalRelabelingStrategyFP1 final: public RelabelingStrategy {
public:
  MultiCausalRelabelingStrategyFP1(size_t response_length,
                                const std::vector<double>& gradient_weights);

  bool relabel(
      const std::vector<size_t>& samples,
      const Data& data,
      Eigen::ArrayXXd& responses_by_sample) const;

  size_t get_response_length() const;

private:
  size_t response_length;
  std::vector<double> gradient_weights;
};

} // namespace grf

#endif //GRF_MULTICAUSALRELABELINGSTRATEGYFP1_H

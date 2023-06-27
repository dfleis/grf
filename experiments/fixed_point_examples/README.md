This folder contains some files serving as a coarse introduction to the fixed-point acceleration of generalized random forests
described in the recent preprint "Accelerating Generalized Random Forests with Fixed-Point Trees" by David Fleischer, 
David A. Stephens, and Archer Yang (2023) https://arxiv.org/abs/2306.11908. Implementations of the acceleration are restricted
to heterogeneous treatment/partial effect estimation for multi-level/continuous treatment variables via the
`multi_arm_causal_forest` and `lm_forest` functions. 

The new argument `method` is available for both `multi_arm_causal_forest` and `lm_forest`, allowing the user to select whether
the forests should be fit using the original gradient-based method (e.g. via the **gradient tree algorithm** of the original grf
specification) or using methods following the **fixed-point tree algorithm** of https://arxiv.org/abs/2306.11908. For both
`multi_arm_causal_forest` and `lm_forest` the argument `method` is a character string of the following form
* `grad`: Original gradient method.
* `fp1`: Exact fixed-point method.
* `fp2`: Approximate fixed-point method for heterogeneous treatment effect estimation for multi-dimensional treatments.

Figures require the tidyverse collection of packages as well as, ggplot2, ggh4x, pals.


Contained in this folder are the following scripts/tests:
* TO DO...
* TO DO...

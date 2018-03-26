# psNMF-LVA2018
This repository contains functions for running Probabilistic Sparse Non-negative Matrix Factorization (psNMF) and its non-sparse counterpart probabilistic NMF.

The aim of NMF is to factorize a data matrix X into two latent matrices W and H, such that X = W*H. This tool facilitates the scenarios where,

1) W and/or H follows a truncated normal distribution.
2) W and/or H follows an exponential distribution.
3) W or H follows a Uniform(0,1) distribution

In addition, for 1) and 2) the scale of the distribution can be either,

a) Fixed to a scalar value

b) Enforce column-wise sparsity (i.e. automatic relevance determination)

c) Enforce element-wise sparsity (i.e. discover an underlying sparse pattern)

## Getting started

1) For a simple demostration, Run 'demo_psNMF.m'
2) For recreating the results shown in [1], run 'demo_reproduce_lva_results.m'. This will take some time to run.
3) For more details read the documentation in 'VB_CP_ALS.m'


## Notes regarding usage.
All code can be used freely in research and other non-profit applications (unless otherwise stated in the third party licenses. If you publish results obtained with this toolbox we kindly ask that our and other relevant sources are properly cited.

This tool was developed at:

The Technical University of Denmark, Department for Applied Mathematics and Computer Science, Section for Cognitive Systems.

## References

[1] Hinrich, J. L., Mørup, M.: "Probabilistic Sparse Non-negative Matrix Factorization" (Under review)

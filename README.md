# KTH Adv Machine Learning Course Final Project DD2434 2019

## Abstract

This work further develops on the Lawrence et al. papers [1] and [2].
We first replicate the algorithm and evaluate it on the same datasets as the original author.
In addition, we apply it to new data and provide a more extensive assessment of its performance against similar methods such as PCA and Kernel PCA.

The original formulation of the Gaussian Process Latent Variable Model (GPLVM) can be regarded as an extension of the Probabilistic Principal Component Analysis (PPCA) introduced by Tipping and Bishop [3] some years earlier. Thus, the GPLVM consists on the introduction of the kernel trick intuition in this probabilistic reformulation of PCA, that leads to a decomposition of the data represented in higher dimensional spaces.

## Results

### High-dimensional data visualization and clustering
Projection of 48-dimensional features into a 2 dimensional latent space of a subset of mice gene dataset with real classes (left) and their corresponding GMM clustering (right). Notice how clustering in GPLVM projection outperforms the one provided by PCA:

![Mice Gene](/scripts/figures/mice/plot.png)

## References

[1] Neil  Lawrence. Probabilistic  non-linear  principal  component  analysis  with  gaussian  process  latent  variablemodels. Journal of machine learning research, 6(Nov):1783–1816, 2005

[2] Neil  D  Lawrence. Gaussian  process  latent  variable  models  for  visualisation  of  high  dimensional  data.   In Advances in neural information processing systems, pages 329–336, 2004.

[3] Michael E Tipping and Christopher M Bishop. Probabilistic principal component analysis. Journal of the
Royal Statistical Society: Series B (Statistical Methodology), 61(3):611–622, 1999.

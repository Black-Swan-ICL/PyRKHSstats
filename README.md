# PyRKHSstats
A Python package implementing a variety of statistical/machine learning methods 
that rely on kernels (e.g. HSIC for independence testing).

## Overview
- Independence testing with HSIC (Hilbert-Schmidt Independence Criterion), as 
  introduced in
  [A Kernel Statistical Test of Independence](https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html), 
  A. Gretton, K. Fukumizu, C. Hui Teo, L. Song, B. Sch&#246;lkopf, and A. 
  Smola (NIPS 2007).
- Measurement of conditional independence with HSCIC (Hilbert-Schmidt 
  Conditional Independence Criterion), as introduced in 
  [A Measure-Theoretic Approach to Kernel Conditional Mean Embeddings](https://papers.nips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html),
  J. Park and K. Muandet (NeurIPS 2020).
- The Kernel-based Conditional Independence Test (KCIT), as introduced in 
  [Kernel-based Conditional Independence Test and Application in Causal 
  Discovery](https://arxiv.org/abs/1202.3775), K. Zhang, J. Peters, D. Janzing,
  B. Sch&#246;lkopf (UAI 2011).
- Two-sample testing (also known as homogeneity testing) with the MMD 
  (Maximum Mean Discrepancy), as presented in [A Fast, Consistent Kernel 
  Two-Sample Test](https://papers.nips.cc/paper/2009/hash/9246444d94f081e3549803b928260f56-Abstract.html),
  A. Gretton, K. Fukumizu, Z. Harchaoui, and B. K. Sriperumbudur (NIPS 2009) 
  and in [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html), 
  A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Sch&#246;lkopf, and A. Smola 
  (JMLR, volume 13, 2012).

<br>

| Resource | Description | 
| :---  | :--- | 
| HSIC | For independence testing | 
| HSCIC | For the measurement of conditional independence | 
| KCIT | For conditional independence testing | 
| MMD | For two-sample testing |


## Implementations available

The following table details the implementation schemes for the different 
resources available in the package.

| Resource | Implementation Scheme | Numpy based available | PyTorch based available |
| :---  | :--- | :----: |:----: |
| HSIC | Resampling (permuting the x<sub>i</sub>'s but leaving the y<sub>i</sub>'s unchanged) | Yes | No |
| HSIC | Gamma approximation | Yes | No |
| HSCIC | N/A | Yes | Yes |
| KCIT | Gamma approximation | Yes | No |
| KCIT | Monte Carlo simulation (weighted sum of &chi;<sup>2</sup> random variables)| Yes | No |
| MMD | Gram matrix spectrum | Yes | No |

[comment]: <> (| MMD | Permutation | Yes | No |)

<br>

## In development
- Joint independence testing with dHSIC.
- Goodness-of-fit testing.
- Methods for time series models.
- Bayesian statistical kernel methods.
- Regression by independence maximisation.
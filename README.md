# PyRKHSstats
A Python package implementing a variety of statistical/machine learning methods 
that rely on kernels (e.g. HSIC for independence testing).

## Implemented
- Independence testing with HSIC (Hilbert-Schmidt Independence Criterion) using
  the Gamma approximation, as introduced in
  [A Kernel Statistical Test of Independence](https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html), 
  A. Gretton, K. Fukumizu, C. Hui Teo, L. Song, B. Scholkopf, and A. J. Smola 
  (NIPS 2007).
- Measurement of conditional independence with HSCIC (Hilbert-Schmidt 
  Conditional Independence Criterion), as introduced in 
  [A Measure-Theoretic Approach to Kernel Conditional Mean Embeddings](https://papers.nips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html),
  J. Park and K. Muandet (NeurIPS 2020).
- The Kernel-based Conditional Independence Test (KCIT), as introduced in 
  [Kernel-based Conditional Independence Test and Application in Causal 
  Discovery](https://arxiv.org/abs/1202.3775), K. Zhang, J. Peters, D. Janzing,
  B. Scholkopf (UAI 2011).

<br>

| Resource | Description | Numpy based available | PyTorch based available |
| :---  | :--- | :----: |:----: |
| HSIC | For independence testing | Yes | No |
| HSCIC | For the measurement of conditional independence | Yes | Yes |
| KCIT | For conditional independence testing | Yes | No |

<br>
  
## In development
- Two-sample testing with MMD.
- Goodness-of-fit testing.
- Methods for time series models.
- Bayesian statistical kernel methods.
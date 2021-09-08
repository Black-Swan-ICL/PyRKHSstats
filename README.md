# PyRKHSstats
A Python package implementing a variety of statistical/machine learning methods 
that rely on kernels (e.g. HSIC for independence testing).

## Implemented
- Independence testing with HSIC (Hilbert-Schmidt Independence Criterion) using
  the Gamma approximation, as introduced in
  'A Kernel Statistical Test of Independence', A. Gretton, K. Fukumizu, C. Hui 
  Teo, L. Song, B. Scholkopf, and A. J. Smola (2007).
- Measurement of conditional independence with HSCIC (Hilbert-Schmidt 
  Conditional Independence Criterion), as introduced in 'A Measure-Theoretic 
  Approach to Kernel Conditional Mean Embeddings', J. Park and K. Muandet 
  (2020).

<br>

| Resource | Description | Numpy based available | PyTorch based available |
| :---  | :--- | :----: |:----: |
| HSIC | For independence testing | Yes | No |
| HSCIC | For the measurement of conditional independence | Yes | Yes |

<br>
  
## In development
- Two-sample testing with MMD.
- Goodness-of-fit testing.
- Methods for time series models.
- Bayesian statistical kernel methods.
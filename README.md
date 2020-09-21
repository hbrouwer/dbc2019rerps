# About

This implements the regression-based waveform estimation (rERP) analysis detailed in:

[Brouwer, H., Delogu, F., and Crocker, M. W. (2020). Splitting Event‐Related Potentials: Modeling Latent Components using Regression‐based Waveform Estimation. *European Journal of Neuroscience*. doi: 10.1111/ejn.14961](https://onlinelibrary.wiley.com/doi/abs/10.1111/ejn.14961)

which is applied to the ERP data reported in:

[Delogu, F., Brouwer, H., and Crocker, M. W. (2019). Event-related potentials index lexical retrieval (N400) and integration (P600) during language comprehension. *Brain and Cognition, 135*. doi: 10.1016/j.bandc.2019.05.007](https://www.sciencedirect.com/science/article/pii/S0278262618304299)

# Getting started

Clone this repository, download the data files (see Releases) and decompress them in the repository folder. 

# Requirements

To run the analysis, you need:

* A recent version of Python 3, with recent versions of:
  * NumPy
  * pandas
  * SciPy
  * Matplotlib
* GNU Make (optional)

# Usage

To build the rERP analysis, including all graphs and time-window averages:

```
$ make analysis
```

To make density plots for the ratings:

```
$ make ratings
```

To undo everything:

```
$ make clean
```

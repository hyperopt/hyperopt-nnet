hyperopt-nnet
=============

This package provides a
[hyperopt](http://jaberg.github.io/hyperopt)-compatible neural network
implementation.

Currently, it can be used to tune neural network hyperparameters for data sets
provided as [skdata](http://jaberg.github.io/skdata) protocols.

See the `./examples` subdirectory for sample training scripts (e.g. `nips2011_nnet.py`)
and a plotting script (`plot_trials.py`).

The `hpnnet.nips2011` file implements the search parameterization used in
Bergstra, Bardenet, Bengio, and Kegl ("[Algorithms for Hyper-parameter
Optimization](http://books.nips.cc/papers/files/nips24/NIPS2011_1385.pdf)") from NIPS 2011.


Dependencies
------------

* NumPy
* Sklearn
* Theano
* Skdata  (github master, not PyPI)
* Hyperopt (github master, not PyPI)
* matplotlib (for plotting)
* IPython (for parallel search, option 1)
* MongoDB (for parallel search, option 2)



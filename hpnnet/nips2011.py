"""
Neural Network (NNet) and Deep Belief Network (DBN) search spaces used in [1]
and [2].

The functions in this file return pyll graphs that can be used as the `space`
argument to e.g. `hyperopt.fmin`. The pyll graphs include hyperparameter
constructs (e.g. `hyperopt.hp.uniform`) so `hyperopt.fmin` can perform
hyperparameter optimization.

See ./skdata_learning_algo.py for example usage of these functions.


[1] Bergstra, J.,  Bardenet, R., Bengio, Y., Kegl, B. (2011). Algorithms
for Hyper-parameter optimization, NIPS 2011.

[2] Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter
Optimization, JMLR 13:281--305.

"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import numpy as np

from hyperopt.pyll import scope
from hyperopt import hp

import pyll_stubs
import nnet  # -- load scope with nnet symbols


def nnet1_preproc_space(sup_min_epochs=300, sup_max_epochs=4000):
    """
    Return a hyperopt-compatible pyll expression for a trained neural network.

    The trained neural network will have one hidden layer, and may
    have an affine first layer that does column normalization or PCA
    pre-processing.

    The training program is built using stub literals `pyll_stubs.train_task`
    and `pyll_stubs.valid_task`.  When evaluating the pyll program, these
    literals must be replaced with skdata Task objects with
    `vector_classification` semantics.  See `skdata_learning_algo.py` for how
    to use the `use_obj_for_literal_in_memo` function to swap live Task
    objects in for these stubs.

    The search space described by this function corresponds to the one-layer
    neural network with pre-processing used in [1] and [2].

    """
    import sys
    print >> sys.stderr, "TODO: l2 penalty"
    print >> sys.stderr, "TODO: PCA eps fixed to 1e-14"
    print >> sys.stderr, "TODO: time_limit"

    nnet0 = scope.NNet([])
    nnet1 = hp.choice('preproc',
        [
            scope.nnet_add_layer(
                nnet0,
                scope.column_normalize_layer(
                    scope.getattr(pyll_stubs.train_task, 'x'),
                    std_thresh=hp.loguniform('colnorm_thresh', -8, -2))),
            scope.nnet_add_layer(
                nnet0,
                scope.pca_layer(
                    scope.getattr(pyll_stubs.train_task, 'x'),
                    energy=hp.uniform('pca_energy', .5, 1),
                    eps=hp.loguniform('pca_eps', np.log(1e-14),
                        np.log(1e-1)))),
        ])
    first_tuned_layer = scope.random_logistic_layer(
            n_in=scope.getattr(nnet1, 'n_out'),
            n_out=hp.qloguniform(
                'nhid1', np.log(16), np.log(2000), q=16),
            dist=hp.choice('dist1', ['uniform', 'normal']),
            scale_heuristic=hp.choice('scale_heur1', [
                ('old', hp.uniform('scale_mult1', .2, 2)),
                ('Glorot', )]),
            seed=hp.choice('iseed', [5, 6, 7, 8]),
            )
    nnet2 = scope.nnet_add_layer(nnet1, first_tuned_layer)
    nnet3 = scope.nnet_add_layer(
        nnet2,
        scope.zero_layer(
            n_in=scope.getattr(nnet2, 'n_out'),
            n_out=scope.getattr(pyll_stubs.train_task, 'n_classes')))

    nnet4 = scope.sgd_finetune(
        nnet3,
        pyll_stubs.train_task,
        pyll_stubs.valid_task,
        first_tuned_layer=first_tuned_layer,
        max_epochs=sup_max_epochs,
        min_epochs=sup_min_epochs,
        batch_size=hp.choice('batch_size', [20, 100]),
        lr=hp.lognormal('lr', np.log(.01), 3.),
        lr_anneal_start=hp.qloguniform(
            'lr_anneal_start', np.log(100), np.log(10000), q=1),
        l2_penalty=hp.choice('lr_penalty', [
            0,
            hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 3.)]),
        )

    return nnet4



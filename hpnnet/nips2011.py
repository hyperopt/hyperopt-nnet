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


def nnet1_preproc_space(sup_min_epochs=300, sup_max_epochs=2000,
                       max_seconds=60 * 60):
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
    time_limit = scope.time() + max_seconds

    train_task_x = scope.getattr(pyll_stubs.train_task, 'x')
    nnet0 = scope.NNet([], n_out=scope.getattr(train_task_x, 'shape')[1])
    nnet1 = hp.choice('preproc',
        [
            # -- raw XXX set up something for n_in arg of hidden layer
            nnet0,
            # -- normalize
            scope.nnet_add_layer(
                nnet0,
                scope.column_normalize_layer(
                    train_task_x,
                    std_thresh=hp.loguniform('colnorm_thresh',
                                             np.log(1e-9),
                                             np.log(1e-3)))),
            # -- pca (with bias to throw away a lot)
            scope.nnet_add_layer(
                nnet0,
                scope.pca_layer(
                    train_task_x,
                    energy=hp.uniform('pca_energy', .5, 1),
                    eps=1e-14)),
        ])
    hidden_layer = scope.random_sigmoid_layer(
            n_in=scope.getattr(nnet1, 'n_out'),
            n_out=hp.qloguniform(
                'nhid1', np.log(16), np.log(1024), q=16),
            dist=hp.choice('dist1', ['uniform', 'normal']),
            scale_heuristic=hp.choice('scale_heur1', [
                ('old', hp.uniform('scale_mult1', .2, 2)),
                ('Glorot', )]),
            seed=hp.choice('iseed', [5, 6, 7, 8]),
            squash=hp.choice('squash', ['tanh', 'logistic']),
            )
    nnet2 = scope.nnet_add_layer(nnet1, hidden_layer)
    nnet3 = scope.nnet_add_layer(
        nnet2,
        scope.zero_softmax_layer(
            n_in=scope.getattr(nnet2, 'n_out'),
            n_out=scope.getattr(pyll_stubs.train_task, 'n_classes')))

    nnet4 = scope.nnet_sgd_finetune_classifier(
        nnet3,
        pyll_stubs.train_task,
        pyll_stubs.valid_task,
        fixed_nnet=nnet1,   # -- don't fine-tune this first part of nnet3
        max_epochs=sup_max_epochs,
        min_epochs=sup_min_epochs,
        batch_size=hp.choice('batch_size', [20, 100]),
        lr=hp.lognormal('lr', np.log(.01), 3.),
        lr_anneal_start=hp.qloguniform(
            'lr_anneal_start', np.log(100), np.log(10000), q=1),
        l2_penalty=hp.choice('l2_penalty', [
            0,
            hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 2.)]),
        time_limit=time_limit,
        )

    return nnet4



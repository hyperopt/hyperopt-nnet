"""
Deep Belief Network (DBN) search spaces used in [1] and [2].

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
from hyperopt.fmin import fmin_pass_expr_memo_ctrl

import pyll_stubs
import nnet  # -- load scope with nnet symbols


def preproc_space(
    sup_min_epochs=300,
    sup_max_epochs=4000,
    # the original timeout was 60 minutes, but computers are faster now
    # so ballparking .... 20 minutes
    max_seconds=20 * 60,
    ):
    """
    Return a hyperopt-compatible pyll expression for a trained neural network.

    The trained neural network will have 0, 1, 2, or 3 hidden layers, and may
    have an affine first layer that does column normalization or PCA
    pre-processing.

    Each layer of the network will be pre-trained by some amount of
    contrastive divergence before being fine-tuning by SGD.

    The training program is built using stub literals `pyll_stubs.train_task`
    and `pyll_stubs.valid_task`.  When evaluating the pyll program, these
    literals must be replaced with skdata Task objects with
    `vector_classification` semantics.  See `skdata_learning_algo.py` for how
    to use the `use_obj_for_literal_in_memo` function to swap live Task
    objects in for these stubs.

    The search space described by this function corresponds to the DBN model
    used in [1] and [2].

    """

    X = scope.getattr(pyll_stubs.train_task, 'x')
    nnet0 = scope.NNet([], n_out=scope.getattr(X, 'shape')[1])
    nnet1 = hp.choice('preproc',
        [
            nnet0,                 # -- raw data
            scope.nnet_add_layer(  # -- ZCA of data
                nnet0,
                # TODO: accept two return layers here
                scope.zca_layer(
                    X,
                    energy=hp.uniform('pca_energy', .5, 1),
                    eps=1e-14,
                    )),
        ])

    # TODO: make layer_transform work on whole nnet??
    # TODO: make this work when nnet1 is actually null nnet0 model
    X = scope.layer_transform(nnet1, X)

    param_seed = hp.choice('iseed', [5, 6, 7, 8])

    time_limit = scope.time() + max_seconds

    nnets = [nnet1]
    nnet_i = nnet1
    for ii, cd_epochs_max in enumerate([3000, 2000, 1500]):
        layer = scope.random_logistic_layer(
            # -- hack to get different seeds for dif't layers
            seed=param_seed + cd_epochs_max,
            n_in=scope.getattr(nnet_i, 'n_out'),
            n_out=hp.qloguniform('n_hid_%i' % ii,
                                 np.log(2**7),
                                 np.log(2**12),
                                 q=16),
            W_init_dist=hp.choice('W_idist_%i' % ii, ['uniform', 'normal']),
            W_init_algo=hp.choice('W_ialgo_%i' % ii, ['old', 'Xavier']),
            # -- multiplier should have been nested to go with algo=='old'
            W_init_algo_old_multiplier=hp.lognormal('W_imult_%i' % ii, 0, 1),
            )
        rbm = scope.layer_pretrain_cd(
            layer,
            X,
            lr=hp.lognormal('cd_lr_%i' % ii, np.log(.01), 2),
            epochs=hp.qloguniform('cd_epochs_%i' % ii,
                                  np.log(1),
                                  np.log(cd_epochs_max),
                                  q=1),
            # -- for whatever reason (?), this was fixed
            batchsize=100,
            sample_v0s=hp.choice('sample_v0s_%i' % ii, [False, True]),
            lr_anneal_start=hp.qloguniform('lr_anneal_%i' % ii,
                                           np.log(10),
                                           np.log(10000),
                                           q=1),
            time_limit=time_limit,
            )
        nnet_i = scope.nnet_add_layer(nnet_i, rbm)
        nnets.append(nnet_i)
        X = scope.layer_transform(rbm, X)

    # this prior is not what I would do now, but it is what I did then...
    nnet_features = hp.pchoice(
        'depth',
        [(.5, nnets[0]),
         (.25, nnets[1]),
         (.125, nnets[2]),
         (.125, nnets[3])])

    sup_nnet = scope.nnet_add_layer(
        nnet_features,
        scope.zero_layer(
            n_in=scope.getattr(nnet_features, 'n_out'),
            n_out=scope.getattr(pyll_stubs.train_task, 'n_classes')))


    nnet4, report = scope.nnet_sgd_finetune(
        sup_nnet,
        pyll_stubs.train_task,
        pyll_stubs.valid_task,
        fixed_nnet=nnet1,
        max_epochs=sup_max_epochs,
        min_epochs=sup_min_epochs,
        batch_size=hp.choice('batch_size', [20, 100]),
        lr=hp.lognormal('lr', np.log(.01), 3.),
        lr_anneal_start=hp.qloguniform(
            'lr_anneal_start',
            np.log(100),
            np.log(10000),
            q=1),
        l2_penalty=hp.choice('lr_penalty', [
            0,
            hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 2.)]),
        time_limit=time_limit,
        )

    return nnet4, report



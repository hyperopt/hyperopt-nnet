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

import pyll_stubs
import nnet  # -- load scope with nnet symbols


def preproc_space(
    sup_min_epochs=300,
    sup_max_epochs=4000,
    max_seconds=60 * 60,
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

    train_task_x = scope.getattr(pyll_stubs.train_task, 'x')
    nnet0 = scope.NNet([], n_out=scope.getattr(train_task_x, 'shape')[1])
    nnet1 = hp.choice('preproc',
        [
            nnet0,                  # -- raw data
            scope.nnet_add_layers(  # -- ZCA of data
                nnet0,
                scope.zca_layer(
                    train_task_x,
                    energy=hp.uniform('pca_energy', .5, 1),
                    eps=1e-14,
                    )),
        ])

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
            dist=hp.choice('W_idist_%i' % ii, ['uniform', 'normal']),
            scale_heuristic=hp.choice(
                'W_ialgo_%i' % ii, [
                    ('old', hp.lognormal('W_imult_%i' % ii, 0, 1)),
                    ('Glorot',)]),
            )
        nnet_i_raw = scope.nnet_add_layer(nnet_i, layer)
        # -- repeatedly calculating lower-layers wastes some CPU, but keeps
        #    memory usage much more stable across jobs (good for cluster)
        #    and the wasted CPU is not so much overall.
        nnet_i_pt = scope.nnet_pretrain_top_layer_cd(
            nnet_i_raw,
            train_task_x,
            lr=hp.lognormal('cd_lr_%i' % ii, np.log(.01), 2),
            seed=hp.randint('cd_seed_%i' % ii, 10),
            n_epochs=hp.qloguniform('cd_epochs_%i' % ii,
                                    np.log(1),
                                    np.log(cd_epochs_max),
                                    q=1),
            # -- for whatever reason (?), this was fixed at 100
            batchsize=100,
            sample_v0s=hp.choice('sample_v0s_%i' % ii, [False, True]),
            lr_anneal_start=hp.qloguniform('lr_anneal_%i' % ii,
                                           np.log(10),
                                           np.log(10000),
                                           q=1),
            time_limit=time_limit,
            )
        nnets.append(nnet_i_pt)

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



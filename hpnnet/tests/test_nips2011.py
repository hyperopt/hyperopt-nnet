from functools import partial
import numpy as np
import hyperopt
from hyperopt import pyll
from hyperopt.fmin import fmin_pass_expr_memo_ctrl

from hpnnet.nips2011 import nnet1_preproc_space
from hpnnet.skdata_learning_algo import PyllLearningAlgo, eval_fn

from skdata.larochelle_etal_2007.view import RectanglesVectorXV
from skdata.larochelle_etal_2007.view \
        import MNIST_RotatedBackgroundImages_VectorXV

def test_nnet_rectangles():

    rectangles_eval_fn = partial(eval_fn,
        protocol_cls=RectanglesVectorXV)

    fmin_pass_expr_memo_ctrl(rectangles_eval_fn)

    trials = hyperopt.Trials()

    hyperopt.fmin(
        rectangles_eval_fn,
        space=nnet1_preproc_space(sup_min_epochs=20, sup_max_epochs=40),
        max_evals=10,
        algo=hyperopt.rand.suggest,
        trials=trials,
        )

def test_nnet_mrbi():

    rectangles_eval_fn = partial(eval_fn,
        protocol_cls=MNIST_RotatedBackgroundImages_VectorXV)

    fmin_pass_expr_memo_ctrl(rectangles_eval_fn)

    trials = hyperopt.Trials()

    hyperopt.fmin(
        rectangles_eval_fn,
        space=nnet1_preproc_space(sup_min_epochs=20, sup_max_epochs=40),
        max_evals=10,
        algo=hyperopt.rand.suggest,
        trials=trials,
        )

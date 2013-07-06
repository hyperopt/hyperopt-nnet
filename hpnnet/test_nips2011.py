from functools import partial
import numpy as np
import hyperopt
from hyperopt import pyll
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from nips2011 import nnet1_space
from nips2011 import PyllLearningAlgo
from nips2011 import eval_fn
# TODO: make this a Protocol
from skdata.larochelle_etal_2007.view import RectanglesVectorXV

def test_nnet_iris():

    rectangles_eval_fn = partial(eval_fn,
        protocol_cls=RectanglesVectorXV)

    fmin_pass_expr_memo_ctrl(rectangles_eval_fn)

    trials = hyperopt.Trials()

    hyperopt.fmin(
        rectangles_eval_fn,
        space=nnet1_space(),
        max_evals=10,
        algo=hyperopt.rand.suggest,
        trials=trials,
        )


from hpnnet import nips2011_dbn
from hyperopt.pyll.stochastic import sample

from functools import partial
#import numpy as np
import hyperopt
#from hyperopt import pyll
from hyperopt.fmin import fmin_pass_expr_memo_ctrl

from hpnnet.skdata_learning_algo import eval_fn

from skdata.larochelle_etal_2007.view import RectanglesVectorXV

def test_preproc_space():
    rectangles_eval_fn = partial(eval_fn,
        protocol_cls=RectanglesVectorXV)

    fmin_pass_expr_memo_ctrl(rectangles_eval_fn)

    trials = hyperopt.Trials()
    space = nips2011_dbn.preproc_space()

    hyperopt.fmin(
        rectangles_eval_fn,
        space=space,
        max_evals=10,
        algo=hyperopt.rand.suggest,
        trials=trials,
        )



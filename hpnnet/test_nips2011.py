import numpy as np
import hyperopt
from hyperopt import pyll
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from nips2011 import nnet1_space
from nips2011 import PyllLearningAlgo
# TODO: make this a Protocol
from skdata.larochelle_etal_2007.view import RectanglesVectorXV

@fmin_pass_expr_memo_ctrl
def eval_fn(expr, memo, ctrl):
    protocol = RectanglesVectorXV()
    algo = PyllLearningAlgo(expr, memo, ctrl)
    protocol.protocol(algo)
    results = algo.results
    valid_losses = []
    for dct in results['best_model']:
        del dct['model'] # -- too big, not worth saving
        valid_losses.append(dct['report']['best_epoch_valid'])

    if valid_losses:
        return {
                'loss': float(np.mean(valid_losses)),
                'status': 'ok',
                'algo_results': results,
                }
    else:
        return {
                'status': 'fail',
                'algo_results': results,
                }

def test_nnet_iris():

    trials = hyperopt.Trials()

    hyperopt.fmin(
        eval_fn,
        space=nnet1_space(),
        max_evals=10,
        algo=hyperopt.rand.suggest,
        trials=trials,
        )



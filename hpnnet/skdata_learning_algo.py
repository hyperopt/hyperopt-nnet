"""
Training and construction routines for neural networks.

"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import numpy as np
from skdata.base import SemanticsDelegator
from hyperopt.utils import use_obj_for_literal_in_memo
from hyperopt.pyll import rec_eval
import pyll_stubs


class PyllLearningAlgo(SemanticsDelegator):
    def __init__(self, expr, memo, ctrl):
        self.expr = expr
        self.memo = dict(memo)
        self.ctrl = ctrl
        self.validation_sets = []
        self.results = {
            'best_model': [],
            'loss': [],
        }


    def best_model_vector_classification(self, train, valid):
        # TODO: use validation set if not-None
        memo = dict(self.memo)
        use_obj_for_literal_in_memo(self.expr, train, pyll_stubs.train_task, memo)
        use_obj_for_literal_in_memo(self.expr, valid, pyll_stubs.valid_task, memo)
        use_obj_for_literal_in_memo(self.expr, self.ctrl, pyll_stubs.ctrl, memo)
        model, report = rec_eval(self.expr, memo=memo)
        if model:
            model.trained_on = train.name
        if valid and valid.name not in self.validation_sets:
            self.validation_sets.append(valid.name)
        self.results['best_model'].append(
            {
                'train_name': train.name,
                'valid_name': valid.name if valid else None,
                'model': model,
                'report': report,
            })
        return model


    def loss_vector_classification(self, model, task):
        if model is None:
            err_rate = 1.0
            self.results['loss'].append(
                {
                    'err_rate': err_rate,
                    'task_name': task.name,
                })
        else:
            p = model.predict(task.x)
            err_rate = np.mean(p != task.y)

            # save as string to save space and maintain
            # readability
            assert np.max(p) < 10
            p_str = ''.join(map(str, p))

            self.results['loss'].append(
                {
                    'model_trained_on': model.trained_on,
                    'predictions': p_str,
                    'err_rate': err_rate,
                    'n': len(p),
                    'task_name': task.name,
                })

        return err_rate


def eval_fn(expr, memo, ctrl, protocol_cls):
    protocol = protocol_cls()
    algo = PyllLearningAlgo(expr, memo, ctrl)
    protocol.protocol(algo)
    results = algo.results
    valid_losses = []
    true_loss = None
    for dct in results['best_model']:
        del dct['model'] # -- too big, not worth saving
        valid_losses.append(dct['report']['best_epoch_valid'])

    for dct in results['loss']:
        if dct['task_name'] == 'test':
            true_loss = dct['err_rate']

    if valid_losses:
        rval = {
                'loss': float(np.mean(valid_losses)),
                'status': 'ok',
                'algo_results': results,
                }
    else:
        rval = {
                'status': 'fail',
                'algo_results': results,
                }
    if true_loss != None:
        rval['true_loss'] = true_loss
        print 'true_loss: ', true_loss
    else:
        print 'No true_loss'

    return rval


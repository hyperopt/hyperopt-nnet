import numpy as np

from skdata.base import SemanticsDelegator

from hyperopt.pyll import scope
from hyperopt.pyll import rec_eval
from hyperopt.pyll import Literal
from hyperopt import hp


_train_task = Literal()
_valid_task = Literal()
_ctrl = Literal()


class PyllLearningAlgo(SemanticsDelegator):

    def __init__(self, expr, memo, ctrl):
        self.expr = expr
        self.memo = dict(memo)
        self.ctrl = ctrl
        self.results = {
            'best_model': [],
            'loss': [],
        }


    def best_model_vector_classification(self, train, valid):
        # TODO: use validation set if not-None
        memo = dict(self.memo)
        memo[_train_task] = train
        memo[_valid_task] = valid
        memo[_ctrl] = self.ctrl
        model = rec_eval(self.expr, memo=memo)
        model.trained_on = train.name
        self.results['best_model'].append(
            {
                'train_name': train.name,
                'valid_name': valid.name if valid else None,
                'model': model,
            })
        return model


    def loss_vector_classification(self, model, task):
        p = model.predict(task.x)
        err_rate = np.mean(p != task.y)

        self.results['loss'].append(
            {
                'model_trained_on': model.trained_on,
                'predictions': p,
                'err_rate': err_rate,
                'n': len(p),
                'task_name': task.name,
            })

        return err_rate


@scope.define
class NNet(object):
    def __init__(self, layers):
        self.layers = layers

    @property
    def n_out(self):
        if not self.layers:
            raise IndexError('no layers')
        return self.layers[-1].W.shape[1]

    @property
    def n_in(self):
        return self.layers[0].W.shape[0]

    def predict(self, X, y):
        for layer in self.layers[:-1]:
            X = layer(X)
        preds = np.argmax(X, axis=1)
        return preds


class Layer(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b


class AffineLayer(object):
    def __call__(self, X):
        return np.dot(X, self.W, self.b)


class LogisticLayer(object):
    def __call__(self, X):
        return 1. / (1 + np.exp(-np.dot(X, self.W, self.b)))


class TanhLayer(object):
    def __call__(self, X):
        return np.tanh(np.dot(X, self.W, self.b))


@scope.define
def nnet_add_layer(nnet, layer):
    return NNet(nnet.layers + [layer])


@scope.define
def pca_layer(nnet, *args, **kwargs):
    raise NotImplementedError()

@scope.define
def column_normalize_layer(nnet, *args, **kwargs):
    raise NotImplementedError()


@scope.define
def random_logistic_layer(nnet, *args, **kwargs):
    raise NotImplementedError()


@scope.define
def zero_layer(nnet, *args, **kwargs):
    raise NotImplementedError()

@scope.define
def sgd_finetune(nnet, *args, **kwargs):
    raise NotImplementedError()


def nnet1_space(
    sup_min_epochs=30, # THESE ARE KINDA SMALL FOR SERIOUS RESULTS
    sup_max_epochs=400):

    nnet0 = scope.NNet([])
    nnet1 = hp.choice('preproc',
        [
            scope.nnet_add_layer(
                nnet0,
                scope.column_normalize_layer(
                    scope.getattr(_train_task, 'x'),
                    var_thresh=hp.loguniform('colnorm_vt', -8, -2))),
            scope.nnet_add_layer(
                nnet0,
                scope.pca_layer(
                    scope.getattr(_train_task, 'x'),
                    energy=hp.uniform('zca_energy', .5, 1))),
        ])
    nnet2 = scope.nnet_add_layer(
        nnet1,
        scope.random_logistic_layer(
            n_in=scope.getattr(nnet1, 'n_out'),
            n_out=hp.qloguniform(
                'nhid1', np.log(16), np.log(2000), q=16),
            dist=hp.choice('dist1', ['uniform', 'normal']),
            scale_heuristic=hp.choice('scale_heur1', ['old', 'Xavier']),
            scale_multiplier=hp.uniform('scale_mult1', .2, 2),
            seed=hp.choice('iseed', [5, 6, 7, 8]),
            ))
    nnet3 = scope.nnet_add_layer(
        nnet2,
        scope.zero_layer(
            n_in=scope.getattr(nnet2, 'n_out'),
            n_out=scope.getattr(_train_task, 'n_classes')))

    nnet4 = scope.sgd_finetune(
        nnet3,
        _train_task,
        _valid_task,
        topmost_layers=2,
        max_epochs=sup_max_epochs,
        min_epochs=sup_min_epochs,
        batchsize=hp.choice('batchsize', [20, 100]),
        lr=hp.lognormal('lr', np.log(.01), 3.),
        lr_anneal_start=hp.qloguniform(
            'lr_anneal_start', np.log(100), np.log(10000), q=1),
        l2_penalty=hp.choice('lr_penalty', [
            0,
            hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 3.)]),
        )

    return nnet4


import copy
import numpy as np
import theano
import theano.tensor as TT

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
        self.validation_sets = []
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
        self.layers = list(layers)

    @property
    def n_out(self):
        if not self.layers:
            raise IndexError('no layers')
        return self.layers[-1].n_out

    @property
    def n_in(self):
        if not self.layers:
            raise IndexError('no layers')
        return self.layers[0].n_in

    def predict(self, X, chunk=500):
        preds = []
        for i in range(0, len(X), chunk):
            Xi = X[i * chunk: (i + 1) * chunk]
            for layer in self.layers[:-1]:
                Xi = layer(Xi)
            preds.append(np.argmax(X, axis=1))
        return preds


class Layer(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b

    @property
    def n_out(self):
        return self.W.shape[1]

    @property
    def n_in(self):
        return self.W.shape[0]


class AffineLayer(Layer):
    def __call__(self, X):
        return np.dot(X, self.W) + self.b

    def theano_compute(self, X, W, b):
        return TT.dot(X, W) + b


class AffineElemwiseLayer(Layer):
    def __call__(self, X):
        return X * self.W + self.b

    def theano_compute(self, X, W, b):
        return X * W + b

    @property
    def n_in(self):
        return self.W.shape[1]


class LogisticLayer(Layer):
    def __call__(self, X):
        return 1. / (1 + np.exp(-np.dot(X, self.W) - self.b))

    def theano_compute(self, X, W, b):
        return 1. / (1 + TT.exp(-TT.dot(X, W) - b))


class TanhLayer(Layer):
    def __call__(self, X):
        return np.tanh(np.dot(X, self.W) + self.b)

    def theano_compute(self, X, W, b):
        return TT.tanh(X * W) + b


@scope.define
def nnet_add_layer(nnet, layer):
    return NNet(nnet.layers + [layer])


@scope.define
def pca_layer(X, energy, eps):
    import pylearn_pca
    (eigvals, eigvecs), centered_trainset = pylearn_pca.pca_from_examples(
            X=X,
            max_energy_fraction=energy)
    eigmean = X[0] - centered_trainset[0]

    W = eigvecs / np.sqrt(eigvals + eps)
    b = -np.dot(eigmean, W)
    print('PCA kept %i of %i components' % (W.shape[1], X.shape[1]))
    return AffineLayer(W, b)


@scope.define
def column_normalize_layer(X, std_thresh):
    mean = np.mean(X, axis=0).reshape((1, X.shape[1]))
    std = np.std(X, axis=0).reshape((1, X.shape[1]))
    return AffineElemwiseLayer(
        W=1. / (std + std_thresh),
        b=-mean)


@scope.define
def random_logistic_layer(n_in, n_out, dist,
    scale_heuristic, seed):

    rng = np.random.RandomState(seed)
    if dist == 'uniform':
        WT = rng.uniform(low=-1, high=1, size=(n_out, n_in))
    elif dist == 'normal':
        WT = rng.randn(n_out, n_in)
    else:
        raise ValueError('W_init_dist', dist)

    # N.B. the weights are transposed so that as the number of hidden units
    # changes,
    # the first hidden units are always the same vectors.  this makes it
    # easier to isolate the effect of random initialization from the other
    # hyperparameters (otherwise changing n_out would be pretty much
    # equivalent to re-seeding).
    W = WT.T.astype('float32')

    if scale_heuristic[0] == 'old':
        W *= scale_heuristic[1] / np.sqrt(n_in)
    elif scale_heuristic[0] == 'Glorot':
        W *= np.sqrt(6.0 / (n_in + n_out))
    else:
        raise ValueError(scale_heuristic)

    b = np.zeros(n_out, dtype='float32')
    return LogisticLayer(W, b)


@scope.define
def zero_layer(n_in, n_out):
    W = np.zeros((n_in, n_out), dtype='float32')
    b = np.zeros(n_out, dtype='float32')
    return LogisticLayer(W, b)


@scope.define
def sgd_finetune(nnet, train_task, valid_task, first_tuned_layer,
        max_epochs, min_epochs, batch_size, lr, lr_anneal_start, l2_penalty):

    layers = nnet.layers

    fixed_layers = layers[:layers.index(first_tuned_layer)]
    tuned_layers = layers[layers.index(first_tuned_layer):]

    # Figure something out for validation
    if valid_task is None:
        from sklearn import cross_validation
        N = len(train_task.x)
        kf = cross_validation.KFold(N, 5)
        train_idxs, valid_idxs = iter(kf).next()

        idxmap = np.random.RandomState(123).permutation(N)

        train_x = train_task.x[idxmap[train_idxs]]
        valid_x = train_task.x[idxmap[valid_idxs]]
        train_y = train_task.y[idxmap[train_idxs]]
        valid_y = train_task.y[idxmap[valid_idxs]]
    else:
        train_x = train_task.x
        valid_x = valid_task.x
        train_y = train_task.y
        valid_y = valid_task.y

    # Filter X through the fixed layers, aka apply
    # pre-processing
    for layer in fixed_layers:
        train_x = layer(train_x)
        valid_x = layer(valid_x)

    train_x = train_x.astype('float32')
    valid_x = valid_x.astype('float32')

    shared_train_x = theano.shared(train_x, borrow=True)
    shared_valid_x = theano.shared(valid_x, borrow=True)
    shared_train_y = theano.shared(train_y, borrow=True)
    shared_valid_y = theano.shared(valid_y, borrow=True)

    batch_idx = TT.iscalar()
    s_lr = TT.fscalar()

    batch_train_x = shared_train_x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_valid_x = shared_valid_x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_train_y = shared_train_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_valid_y = shared_valid_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    params = []
    Ws = []
    bs = []

    for layer in tuned_layers:
        s_W = theano.shared(layer.W)
        s_b = theano.shared(layer.b)
        batch_train_x = layer.theano_compute(batch_train_x, s_W, s_b)
        batch_valid_x = layer.theano_compute(batch_valid_x, s_W, s_b)
        Ws.append(s_W)
        bs.append(s_b)

    train_probs = TT.nnet.softmax(batch_train_x)
    train_loss = TT.mean(
        TT.nnet.categorical_crossentropy(train_probs, batch_train_y))
    params = Ws + bs
    gparams = TT.grad(train_loss, params)
    updates = [(p, p - s_lr * gp) for (p, gp) in zip(params, gparams)]
    train_fn = theano.function([batch_idx, s_lr], train_loss,
            updates=updates)

    valid_err_rate = TT.mean(
            TT.neq(batch_valid_y, TT.argmax(batch_valid_x, axis=1)))
    valid_err_rate_fn = theano.function([batch_idx], valid_err_rate)

    report = {}
    report['best_epoch'] = -1
    report['best_epoch_valid'] = 1.0
    report['best_epoch_train'] = 1.0
    report['best_epoch_test'] = 1.0
    report['status'] = 'ok'
    valid_rate=float('inf')
    test_rate=-float('inf')
    train_rate=-float('inf')

    n_train_batches = len(train_x) // batch_size
    n_valid_batches = len(valid_x) // batch_size

    for epoch in xrange(max_epochs):
        valid_rate = float(np.mean([valid_err_rate_fn(i)
            for i in range(n_valid_batches)]))
        valid_rate_std_thresh = 0.5 * np.sqrt(valid_rate *
                (1 - valid_rate) / (n_valid_batches * batch_size))

        if valid_rate < (report['best_epoch_valid'] - valid_rate_std_thresh):
            report['best_epoch'] = epoch
            report['best_epoch_test'] = test_rate
            report['best_epoch_valid'] = valid_rate
            report['best_epoch_train'] = train_rate
            best_params = copy.deepcopy(params)

        e_lr = lr
        e_lr *= min(1, lr_anneal_start / float(epoch + 1))

        print('Epoch=%i best epoch %i valid %f test %f best_epoch_train %f prev_train %f'%(
            epoch, report['best_epoch'], report['best_epoch_valid'],
            report['best_epoch_test'],
            report['best_epoch_train'], train_rate))

        if epoch > max(min_epochs, 2 * report['best_epoch']):
            break
        train_rate = float(np.mean([train_fn(i, e_lr) for i in
            range(n_train_batches)]))
        if not np.isfinite(train_rate):
            report['status'] = 'fail'
            report['status_info'] = 'train_rate %f' % train_rate
            return None, report

    if report['best_epoch'] >= 0:
        best_nnet = NNet(list(fixed_layers))
        best_Ws = best_params[:len(Ws)]
        best_bs = best_params[len(Ws):]
        for tuned, W, b in zip(tuned_layers, best_Ws, best_bs):
            best_nnet.layers.append(
                tuned.__class__(
                    W.get_value(),
                    b.get_value()))
        return best_nnet, report
    else:
        report['status'] = 'fail'
        report['status_info'] = 'noprog'
        return None, report


def nnet1_space(
    sup_min_epochs=300,
    sup_max_epochs=4000):

    nnet0 = scope.NNet([])
    nnet1 = hp.choice('preproc',
        [
            scope.nnet_add_layer(
                nnet0,
                scope.column_normalize_layer(
                    scope.getattr(_train_task, 'x'),
                    std_thresh=hp.loguniform('colnorm_thresh', -8, -2))),
            scope.nnet_add_layer(
                nnet0,
                scope.pca_layer(
                    scope.getattr(_train_task, 'x'),
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
            n_out=scope.getattr(_train_task, 'n_classes')))

    nnet4 = scope.sgd_finetune(
        nnet3,
        _train_task,
        _valid_task,
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


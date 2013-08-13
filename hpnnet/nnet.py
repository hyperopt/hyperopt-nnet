"""
Training and construction routines for neural networks.

"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import copy
import time as time_module
import numpy as np
import theano
import theano.tensor as TT

from hyperopt.pyll import scope

# TODO: move this to hyperopt.pyll
@scope.define
def time():
    return time_module.time()


@scope.define
class NNet(object):
    def __init__(self, layers, n_out=None):
        self.layers = list(layers)
        self._n_out = n_out

    @property
    def n_out(self):
        if not self.layers:
            if self._n_out is None:
                raise IndexError('n_out: no layers')
            else:
                return self._n_out
        return self.layers[-1].n_out

    @property
    def n_in(self):
        if not self.layers:
            raise IndexError('n_in: no layers')
        return self.layers[0].n_in

    def predict(self, X, chunk=256):
        preds = []
        for i in range(0, len(X), chunk):
            t0 = time_module.time()
            Xi = X[i: i + chunk]
            for layer in self.layers:
                Xi = layer(Xi)
            preds.extend(np.argmax(Xi, axis=1))
            t1 = time_module.time()
            if t1 - t0 > .1:
                print 'WARNING: predicting single chunk took', (t1 - t0)
        assert len(preds) == len(X), (len(preds), len(X))
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
        return 1. / (1. + np.exp(-np.dot(X, self.W) - self.b))

    def theano_compute(self, X, W, b):
        return 1. / (1. + TT.exp(-TT.dot(X, W) - b))


class TanhLayer(Layer):
    def __call__(self, X):
        return np.tanh(np.dot(X, self.W) + self.b)

    def theano_compute(self, X, W, b):
        return TT.tanh(X * W) + b


class ClipLayer(Layer):
    def __call__(self, X):
        tmp = np.dot(X, self.W) + self.b
        return np.clip(tmp, 0, 1)


@scope.define
def layer_transform(layer, X):
    return layer(X)


@scope.define
def nnet_transform(nnet, X):
    for layer in nnet.layers:
        X = layer(X)
    return X


@scope.define
def nnet_add_layer(nnet, layer):
    return NNet(nnet.layers + [layer])

@scope.define
def nnet_add_layers(nnet, layers):
    return NNet(nnet.layers + list(layers))


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
def zca_layer(X, energy, eps):
    import pylearn_pca
    (eigvals,eigvecs), centered_trainset = pylearn_pca.pca_from_examples(
            X=X,
            max_energy_fraction=energy)
    #eigmean = X[0] - centered_trainset[0]

    W = eigvecs / np.sqrt(eigvals + eps)
    #b = -np.dot(eigmean, W)
    print('ZCA kept %i of %i components' % (W.shape[1], X.shape[1]))
    # TODO: verify that this is actually the right algorithm
    l0 = AffineLayer(W, 0)
    l1 = ClipLayer(W.T, 0)
    return [l0, l1]

@scope.define
def column_normalize_layer(X, std_thresh):
    mean = np.mean(X, axis=0).reshape((1, X.shape[1]))
    std = np.std(X, axis=0).reshape((1, X.shape[1]))
    return AffineElemwiseLayer(
        W=1. / (std + std_thresh),
        b=-mean)

@scope.define
def layer_pretrain_cd(layer,
                      X, 
                      lr,
                      epochs,
                      batchsize,
                      sample_v0s, 
                      lr_anneal_start,
                      time_limit=None):
    import sys
    print >> sys.stderr, "ERROR: CD Not Implemented"
    return layer


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


@scope.define_info(o_len=2)
def nnet_sgd_finetune(nnet, train_task, valid_task, fixed_nnet,
    max_epochs, min_epochs, batch_size, lr, lr_anneal_start, l2_penalty,
    time_limit=None):

    import sys
    print >> sys.stderr, "TODO: implement time_limit"

    layers = nnet.layers

    fixed_layers = [l for l in nnet.layers if l in fixed_nnet.layers]
    tuned_layers = [l for l in nnet.layers if l not in fixed_nnet.layers]

    # we need all the fixed layers to precede all the tuned layers
    assert layers[:len(fixed_layers)] == fixed_layers

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

    l2_cost = 0

    for layer in tuned_layers:
        s_W = theano.shared(layer.W)
        s_b = theano.shared(layer.b)
        batch_train_x = layer.theano_compute(batch_train_x, s_W, s_b)
        batch_valid_x = layer.theano_compute(batch_valid_x, s_W, s_b)
        Ws.append(s_W)
        bs.append(s_b)
        l2_cost = l2_cost + (s_W ** 2).sum()

    train_probs = TT.nnet.softmax(batch_train_x)
    train_loss = TT.mean(
        TT.nnet.categorical_crossentropy(train_probs, batch_train_y))
    regularized_loss = train_loss + l2_penalty * l2_cost
    params = Ws + bs
    gparams = TT.grad(regularized_loss, params)
    updates = [(p, p - s_lr * gp) for (p, gp) in zip(params, gparams)]
    train_fn = theano.function(
        [batch_idx, s_lr], regularized_loss,
        updates=updates,
        allow_input_downcast=True)

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



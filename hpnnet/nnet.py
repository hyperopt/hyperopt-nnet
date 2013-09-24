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
try:
    # -- TODO: only import this if we intend to use Theano's GPU codegen
    raise ImportError()
    import theano.sandbox.cuda.rng_curand
    RandomStreams = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams
except ImportError:
    try:
        import theano.sandbox.rng_mrg
        RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
    except ImportError:
        RandomStreams = TT.shared_randomstreams.RandomStreams

from hyperopt.pyll import scope


_x = theano.tensor.dmatrix()
_v = theano.tensor.dmatrix()
softmax = theano.function([_x], theano.tensor.nnet.softmax(_x))

dot_mm = theano.function([_x, _v], theano.tensor.dot(_x, _v))

def np_dot(a, b):
    return dot_mm(a, b)


class DivergenceError(Exception):
    """An iterative numerical algorithm diverged (step size too large).
    """


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
                print 'ETA = %s' % ((len(X) - i) / 256. *  (t1 - t0))
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
        return np_dot(X, self.W) + self.b

    def theano_compute(self, X, W, b):
        return TT.dot(X, W) + b


class AffineLayerPre(Layer):
    def __call__(self, X):
        return np_dot(X + self.b, self.W)

    def theano_compute(self, X, W, b):
        return TT.dot(X + b, W)


class AffineElemwiseLayer(Layer):
    def __call__(self, X):
        return X * self.W + self.b

    def theano_compute(self, X, W, b):
        return X * W + b

    @property
    def n_in(self):
        assert self.W.shape[0] == 1
        return (self.W + self.b).shape[1]


class LogisticLayer(Layer):
    def __call__(self, X):
        return 1. / (1. + np.exp(-np_dot(X, self.W) - self.b))

    def theano_compute(self, X, W, b):
        return 1. / (1. + TT.exp(-TT.dot(X, W) - b))


class SoftmaxLayer(Layer):
    def __call__(self, X):
        return softmax(np_dot(X, self.W) + self.b)

    def theano_compute(self, X, W, b):
        return TT.nnet.softmax(TT.dot(X, W) + b)


class TanhLayer(Layer):
    def __call__(self, X):
        return np.tanh(np_dot(X, self.W) + self.b)

    def theano_compute(self, X, W, b):
        return TT.tanh(TT.dot(X, W) + b)


class ClipLayer(Layer):
    def __call__(self, X):
        tmp = np_dot(X, self.W) + self.b
        return np.clip(tmp, 0, 1)

    def theano_compute(self, X, W, b):
        tmp = theano.dot(X, self.W) + self.b
        rval = TT.clip(tmp, 0, 1)
        assert rval.dtype == X.dtype
        return rval



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
    centering_offset = centered_trainset[0] - X[0]

    W = eigvecs / np.sqrt(eigvals + eps)
    print('PCA kept %i of %i components' % (W.shape[1], X.shape[1]))
    return AffineLayerPre(
        W.astype(X.dtype),
        centering_offset.astype(X.dtype))


@scope.define
def zca_layer(X, energy, eps):
    """
    Return a pair of layers whose output when filtering X will be X's ZCA.

    energy - retain at least this much energy with the principle components
    eps - add this to the eigenvalues when computing PCA responses to prevent
          division-by-zero and suppress weak components in the PCA
          representation.
    """
    import pylearn_pca
    (eigvals, eigvecs), centered_trainset = pylearn_pca.pca_from_examples(
            X=X,
            max_energy_fraction=energy)

    centering_offset = centered_trainset[0] - X[0]
    W = eigvecs / np.sqrt(eigvals + eps)
    print('ZCA kept %i of %i components' % (W.shape[1], X.shape[1]))
    l0 = AffineLayerPre(W.astype(X.dtype), centering_offset.astype(X.dtype))
    l1 = ClipLayer(eigvecs.T.copy().astype(X.dtype), np.asarray(0, dtype=X.dtype))
    return [l0, l1]


@scope.define
def column_normalize_layer(X, std_thresh):
    mean = np.mean(X, axis=0).reshape((1, X.shape[1]))
    std = np.std(X, axis=0).reshape((1, X.shape[1]))
    return AffineElemwiseLayer(
        W=1. / (std + std_thresh),
        b=-mean)


@scope.define
def nnet_pretrain_top_layer_cd(nnet,
                               X, 
                               lr,
                               n_epochs,
                               seed,
                               batchsize,
                               sample_v0s, 
                               lr_anneal_start,
                               time_limit=None):
    """
    Return a new pre-trained version of Layer, trained by contrastive
    divergence.  This is not stochastic maximum-likelihood or persistive CD,
    this is the so-called "CD-1" algorithm.
    """
    dtype = str(X.dtype)
    s_rng = RandomStreams(int(seed))
    s_features = theano.shared(X, borrow=True)
    s_batchsize = TT.as_tensor_variable(batchsize)
    s_idx = TT.lscalar()
    s_lr = theano.shared(np.asarray(lr, dtype=X.dtype))
    if not nnet.layers:
        raise ValueError('nnet_pretrain_top_layer_cd:'
                         ' at least one layer required')
    v0m = s_features[s_idx * s_batchsize: (s_idx + 1) * s_batchsize]
    # -- filter features through lowermost layers
    for layer in nnet.layers[:-1]:
        s_W = theano.shared(layer.W)
        s_b = theano.shared(layer.b)
        tmp = layer.theano_compute(v0m, s_W, s_b)
        assert tmp.dtype == v0m.dtype, layer
        v0m = tmp

    # -- start CD on top layer
    if not isinstance(nnet.layers[-1], LogisticLayer):
        raise TypeError('CD pretraining only works for'
                        ' nnets with Logistic top layer')
    n_in = nnet.layers[-1].n_in
    n_out = nnet.layers[-1].n_out
    print('rbm training n_in=%i n_out=%i batchsize=%i' % (
        n_in, n_out, batchsize))
    s_W = theano.shared(nnet.layers[-1].W)
    s_b = theano.shared(nnet.layers[-1].b)
    s_a = theano.shared(np.zeros(n_in, dtype=s_b.dtype))
    if str(X.dtype) != str(s_W.dtype):
        raise TypeError('data and W have different dtypes')
    if sample_v0s:
        v0s = TT.cast(
                v0m > s_rng.uniform(
                    size=(batchsize, nnet.layers[-1].n_in)),
                dtype)
    else:
        v0s = v0m

    h0m = TT.nnet.sigmoid(TT.dot(v0s, s_W) + s_b)
    h0s = TT.cast(s_rng.uniform(size=(batchsize, n_out)) < h0m, dtype)
    v1m = TT.nnet.sigmoid(TT.dot(h0s, s_W.T) + s_a)
    v1s = TT.cast(s_rng.uniform(size=(batchsize, n_in)) < v1m, dtype)
    h1m = TT.nnet.sigmoid(TT.dot(v1s, s_W) + s_b)

    # -- compile CD1 update function
    cd1_fn = theano.function([s_idx],
            [abs(v0m - v1m).mean()],
            updates=[
                (s_W, s_W + s_lr * (
                    TT.dot(v0s.T, h0m) - TT.dot(v1s.T, h1m))),
                (s_a, s_a + s_lr * (
                    (v0s - v1s).sum(axis=0))),
                (s_b, s_b + s_lr * (
                    (h0m - h1m).sum(axis=0))),
                ],
            )
    n_batches_per_epoch = len(X) / batchsize
    if len(X) > (batchsize * n_batches_per_epoch):
        n_batches_per_epoch += 1
    for epoch in xrange(int(n_epochs)):
        if time_limit and time_module.time() > time_limit:
            break
        e_lr = lr * min(1, (float(lr_anneal_start) / (epoch + 1)))
        s_lr.set_value(float(e_lr))

        costs = [cd1_fn(bi) for bi in xrange(n_batches_per_epoch)]
        if not epoch % 10:
            print('CD1 epoch:%i  avg L1: %f'% (epoch, np.mean(costs)))
        if not np.isfinite(np.mean(costs)):
            raise DivergenceError('CD went crazy')

    new_top_layer = LogisticLayer(W=s_W.get_value(borrow=True),
                                  b=s_b.get_value(borrow=True))
    new_nnet = NNet(nnet.layers[:-1] + [new_top_layer])
    return new_nnet


@scope.define
def random_sigmoid_layer(n_in, n_out, dist,
    scale_heuristic, seed, squash,
    dtype='float32'):

    rng = np.random.RandomState(seed)
    if dist == 'uniform':
        WT = rng.rand(n_out, n_in) * 2 - 1
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
    W = WT.T.astype(dtype)

    if scale_heuristic[0] == 'old':
        W *= scale_heuristic[1] / np.sqrt(n_in)
    elif scale_heuristic[0] == 'Glorot':
        W *= np.sqrt(6.0 / (n_in + n_out))
    else:
        raise ValueError(scale_heuristic)

    b = np.zeros(n_out, dtype=dtype)
    if squash == 'logistic':
        return LogisticLayer(W, b)
    elif squash == 'tanh':
        return TanhLayer(W, b)
    else:
        raise NotImplementedError('squashing function', squash)


@scope.define
def zero_softmax_layer(n_in, n_out, dtype='float32'):
    W = np.zeros((n_in, n_out), dtype=dtype)
    b = np.zeros(n_out, dtype=dtype)
    return SoftmaxLayer(W, b)


@scope.define_info(o_len=2)
def nnet_sgd_finetune_classifier(nnet, train_task, valid_task, fixed_nnet,
    max_epochs, min_epochs, batch_size, lr, lr_anneal_start, l2_penalty,
    time_limit=None, dtype='float32'):

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

    train_x = train_x.astype(dtype)
    valid_x = valid_x.astype(dtype)

    shared_train_x = theano.shared(train_x, borrow=True)
    shared_valid_x = theano.shared(valid_x, borrow=True)
    shared_train_y = theano.shared(train_y, borrow=True)
    shared_valid_y = theano.shared(valid_y, borrow=True)

    batch_idx = TT.iscalar()
    s_lr = TT.scalar(dtype=dtype)

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
        #batch_train_x = theano.printing.Print('x')(batch_train_x)

    # -- the topmost layer is the classifier, so at this point batch_train_x
    #    represents the softmax classifier output.
    train_probs = batch_train_x
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

    # -- the topmost layer is the classifier, so at this point batch_valid_x
    #    represents the softmax classifier output.
    valid_err_rate = TT.mean(
            TT.neq(batch_valid_y, TT.argmax(batch_valid_x, axis=1)))
    valid_err_rate_fn = theano.function([batch_idx], valid_err_rate)

    report = {}
    report['best_epoch'] = -1
    report['best_epoch_valid'] = 1.0
    report['best_epoch_avg_train_reg_loss'] = 1.0
    report['best_epoch_test'] = 1.0
    report['status'] = 'ok'
    valid_err_rate = float('inf')
    test_err_rate = float('inf')
    avg_regularized_loss = float('inf')

    n_train_batches = len(train_x) // batch_size
    n_valid_batches = len(valid_x) // batch_size

    for epoch in xrange(max_epochs):
        valid_err_rate = float(np.mean([valid_err_rate_fn(i)
            for i in range(n_valid_batches)]))
        valid_err_rate_std_thresh = 0.5 * np.sqrt(valid_err_rate *
                (1 - valid_err_rate) / (n_valid_batches * batch_size))

        if valid_err_rate < (
                report['best_epoch_valid'] - valid_err_rate_std_thresh):
            report['best_epoch'] = epoch
            report['best_epoch_test'] = test_err_rate
            report['best_epoch_valid'] = valid_err_rate
            report['best_epoch_avg_train_reg_loss'] = avg_regularized_loss
            best_params = copy.deepcopy(params)

        e_lr = lr
        e_lr *= min(1, lr_anneal_start / float(epoch + 1))

        print('Epoch=%i best epoch %i valid %f test %f '
                ' best_train %f cur_train %f lr %f' % (
            epoch, report['best_epoch'],
            report['best_epoch_valid'],
            report['best_epoch_test'],
            report['best_epoch_avg_train_reg_loss'],
            avg_regularized_loss,
            e_lr))

        if epoch > max(min_epochs, 2 * report['best_epoch']):
            break
        if time_limit is not None and time_module.time() > time_limit:
            break
        # -- loop comprehension does one epoch of training
        avg_regularized_loss = float(np.mean([train_fn(i, e_lr) for i in
            range(n_train_batches)]))
        if not np.isfinite(avg_regularized_loss):
            report['status'] = 'fail'
            report['status_info'] = ('avg_regularized_loss %f' %
                avg_regularized_loss)
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
        print 'nnet_sgd_finetune: ', report
        return best_nnet, report
    else:
        report['status'] = 'fail'
        report['status_info'] = 'noprog'
        return None, report



import sys

class ObjectId(object):
    def __init__(self, _id):
        self._id = _id

class ISODate(object):
    def __init__(self, datestring):
        self.datestring = datestring

config1 = { "_id" : ObjectId("4cd9e13e8a077c67400000d6"), "argd" : { "pca_energy" :
    0.99, "preprocessing" : "pca", "W_init_algo" : "old",
    "W_init_algo_old_multiplier" : 1.121444322228466, "dataset_name" :
    "rectangles", "sup_max_epochs" : 4000, "sup_min_epochs" : 300, "squash" :
    "sigmoid", "iseed" : 5, "batchsize" : 100, "n_hid" : 61, "lr" :
    0.003537415031816308, "lr_anneal_start" : 13768, "l2_penalty" :
    0.000015964911748563186 }, "book_time" :
    ISODate("2010-11-10T00:04:56.467Z"), "cmd" : "nnet_fn_1", "owner" :
    "brams0a.iro.umontreal.ca:13364", "refresh_time" :
    ISODate("2010-11-10T00:05:15.983Z"), "result" : { "best_epoch_test" :
        0.5289400000000007, "best_epoch" : 127, "best_epoch_valid" :
        0.5700000000000001, "version" : 1, "best_epoch_train" :
        0.7132604202506101 }, "status" : 2, "version" : 605 }


config2 = { "_id" : ObjectId("4cd9e13e8a077c6740000021"), "argd" : { "pca_energy" :
    0.99, "preprocessing" : "raw", "W_init_algo" : "old",
    "W_init_algo_old_multiplier" : 0.9229159592523637, "dataset_name" :
    "rectangles", "sup_max_epochs" : 4000, "sup_min_epochs" : 300,
    "squash" : "sigmoid", "iseed" : 5, "batchsize" : 20, "n_hid" : 917,
    "lr" : 6.610082599376332, "lr_anneal_start" : 21782, "l2_penalty" :
    0.000017125453152969375 }, "book_time" :
    ISODate("2010-11-10T00:11:47.329Z"), "cmd" : "nnet_fn_1", "owner" :
    "maggie26.iro.umontreal.ca:544", "refresh_time" :
    ISODate("2010-11-10T00:20:21.452Z"), "result" : { "best_epoch_test" :
        0.93328, "best_epoch" : 232, "best_epoch_valid" : 0.965, "version"
        : 1, "best_epoch_train" : 0.0009210755676031113 }, "status" : 2,
    "version" : 933 }

import hyperopt.pyll
from hyperopt.pyll_utils import expr_to_config
from hpnnet.nips2011 import nnet1_preproc_space

from hpnnet.skdata_learning_algo import eval_fn
from skdata.larochelle_etal_2007.view import RectanglesVectorXV

def run_config(config):
    argd = config['argd']
    def config_lookup(key):
        if key == 'scale_mult1':
            return argd['W_init_algo_old_multiplier']

        if key == 'scale_heur1':
            if 'old' == argd['W_init_algo']:
                return 0
            else:
                assert 'Xavier' == argd['W_init_algo']
                return 1

        if key == 'preproc':
            return {'raw': 0, 'normalize': 1, 'pca': 2}[
                    argd['preprocessing']]

        if key == 'batch_size':
            return 0 if 20 == argd['batchsize'] else 1

        if key == 'nhid1':
            return argd['n_hid']

        if key == 'dist1':
            return 0 if argd['W_init_algo'] == 'old' else 1

        if key == 'squash':
            return 0 if argd['squash'] == 'tanh' else 1

        if key == 'colnorm_thresh':
            return 1e-7

        if key == 'l2_penalty_nz':
            return argd['l2_penalty']

        if key == 'l2_penalty':
            return 0 if argd['l2_penalty'] == 0 else 1

        if key == 'iseed':
            # convert from seed value to choice index
            return argd['iseed'] - 5

        try:
            return argd[key]
        except KeyError:
            print 'Returning GarbageCollected for %s' % key
            return hyperopt.pyll.base.GarbageCollected

    expr = nnet1_preproc_space()
    hps = {}
    expr_to_config(expr, None, hps)
    print config
    memo = {}
    for k, v in hps.items():
        #print k, v
        memo[v['node']] = config_lookup(k)

    print memo
    rval = eval_fn(
        expr=expr,
        memo=memo,
        ctrl=None,
        protocol_cls=RectanglesVectorXV)
    print '-' * 80
    print 'COMPUTED RESULTS IN TERMS OF *ERROR*'
    print rval['loss']
    print '-' * 80
    print 'SAVED RESULTS IN TERMS OF *ACCURACY*'
    print config['result']
    print '-' * 80


if __name__ == '__main__':
    sys.exit(run_config(config1))
    #sys.exit(run_config(config2))


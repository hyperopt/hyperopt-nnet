#!/usr/bin/env python
"""
Evaluating one-layer neural networks on data sets from skdata.

Run this script like e.g.

    $ ipcluster start                 # in shell 1
    $ ./nips2011_nnet1.py rectangles  # in shell 2

This will conduct an ipython-based parallel search for the best
neural network for the "rectangles" data set, and store the results
as a Trials object called "iptrials_rectangles.pkl".  You can plot
the results by typing for example:

    $ ./plot_trials.py iptrials_rectangles.pkl

"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import cPickle
from functools import partial
import sys

from IPython.parallel import Client
from hyperopt import tpe
from hyperopt.ipy import IPythonTrials
from hpnnet.skdata_learning_algo import eval_fn
from hpnnet.nips2011_dbn import preproc_space

def get_iptrials(filename):
    client = Client()
    try:
        iptrials = cPickle.load(open(filename))
        iptrials._client = client
    except IOError:
        iptrials = IPythonTrials(client)
    except (EOFError, cPickle.PickleError):
        print "ERROR: unpickling FAILED"
        iptrials = IPythonTrials(client)
    return iptrials


def main_rectangles(filename='iptrials_rectangles_dbn.pkl'):
    from skdata.larochelle_etal_2007.view import RectanglesVectorXV
    iptrials = get_iptrials(filename)

    rectangles_eval_fn = partial(eval_fn,
        protocol_cls=RectanglesVectorXV)

    for max_evals in [10, 25, 50]:
        iptrials.fmin(
            fn=rectangles_eval_fn,
            space=preproc_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            verbose=1,
            pass_expr_memo_ctrl=True,
            )
        iptrials.wait()
        iptrials.refresh()
        ofile = open(filename, 'w')
        cPickle.dump(iptrials, ofile)
        ofile.close()


def main_MRBI(filename='iptrials_MRBI_dbn.pkl'):
    from skdata.larochelle_etal_2007.view \
            import MNIST_RotatedBackgroundImages_VectorXV as Protocol
    iptrials = get_iptrials(filename)

    dataset_eval_fn = partial(eval_fn, protocol_cls=Protocol)

    for max_evals in range(20, 100, 200):
        iptrials.fmin(
            fn=dataset_eval_fn,
            space=preproc_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            verbose=1,
            pass_expr_memo_ctrl=True,
            )
        iptrials.wait()
        iptrials.refresh()
        ofile = open(filename, 'w')
        cPickle.dump(iptrials, ofile)
        ofile.close()


def main_convex(filename='iptrials_convex_dbn.pkl'):
    from skdata.larochelle_etal_2007.view import ConvexVectorXV as Protocol
    iptrials = get_iptrials(filename)

    dataset_eval_fn = partial(eval_fn,
        protocol_cls=Protocol)

    for max_evals in range(10, 50, 10):
        iptrials.fmin(
            fn=dataset_eval_fn,
            space=preproc_space(),
            algo=tpe.suggest,
            max_evals=max_evals,
            verbose=1,
            pass_expr_memo_ctrl=True,
            )
        iptrials.wait()
        iptrials.refresh()
        ofile = open(filename, 'w')
        cPickle.dump(iptrials, ofile)
        ofile.close()


def main():
    cmd = 'main_' + sys.argv[1]
    main_fn = globals()[cmd]
    return main_fn(*sys.argv[2:])

if __name__ == '__main__':
    sys.exit(main())


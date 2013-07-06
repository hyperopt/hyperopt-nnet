import cPickle
from functools import partial

from IPython.parallel import Client
from hyperopt import tpe, rand
from hyperopt.ipy import IPythonTrials
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from hpnnet.nips2011 import eval_fn
from hpnnet.nips2011 import nnet1_space
from hpnnet.nips2011 import PyllLearningAlgo
from skdata.larochelle_etal_2007.view import RectanglesVectorXV

client = Client()
try:
    iptrials = cPickle.load(open('rectangles.pkl'))
    iptrials._client = client
except IOError:
    iptrials = IPythonTrials(client)
except (EOFError, cPickle.PickleError):
    print "ERROR: unpickling FAILED"
    iptrials = IPythonTrials(client)


rectangles_eval_fn = partial(eval_fn,
    protocol_cls=RectanglesVectorXV)

for max_evals in range(10, 50, 10):
    iptrials.fmin(
            fn=rectangles_eval_fn,
            space=nnet1_space(
                sup_min_epochs=30,
                sup_max_epochs=50,
                ),
            algo=tpe.suggest,
            max_evals=max_evals, verbose=1)
    iptrials.wait()
    iptrials.refresh()
    ofile = open('rectangles.pkl', 'w')
    cPickle.dump(iptrials, ofile)
    ofile.close()


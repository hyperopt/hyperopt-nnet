"""
Singleton objects that serve as placeholders in pyll graphs.

These are used by e.g. ./nips2011.py
"""

class train_task(object):
    """`train` argument to skdata.LearningAlgo's best_model method
    """

class valid_task(object):
    """`valid` argument to skdata.LearningAlgo's best_model method
    """

class ctrl(object):
    """Hyperopt Ctrl object passed to worker eval_fn.
    """



import cPickle
import sys
from hyperopt.plotting import main_plot_history

trials = cPickle.load(open(sys.argv[1]))

main_plot_history(trials)


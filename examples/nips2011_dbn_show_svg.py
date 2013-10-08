"""
To see the relationship between hyperparameters in the nips2011_dbn space:

python nips2011_dbn_show_svg.py && dot -Tpng dbn.dot > dbn.png && eog dbn.png

"""

from hpnnet.nips2011_dbn import preproc_space
from hyperopt.graphviz import dot_hyperparameters
open('dbn.dot', 'wb').write(dot_hyperparameters(preproc_space()))


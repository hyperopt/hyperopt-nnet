import skdata.svhn.view
from hpnnet import pylearn_pca
import matplotlib.pyplot as plt

SHOW = False

def test_zca():
    view = skdata.svhn.view.CroppedDigitsStratifiedKFoldView1()

    print view.splits[0].train.x.shape

    N = 1000

    first_x = view.splits[0].train.x[:N].reshape(N, -1)

    eigstuff, centered_x = pylearn_pca.pca_from_examples(first_x, max_energy_fraction=.99)
    zca_x = pylearn_pca.zca_whiten(eigstuff, centered_x)
    offset = centered_x[0] - first_x[0]
    print zca_x[0].min()
    print zca_x[1].max()
    assert zca_x[0].min()  > -.5
    assert zca_x[0].max()  < 2

    for i in range(4):
        plt.subplot(4, 2, 2 * i + 1)
        print 'first_x', first_x[i].min()
        print 'first_x', first_x[i].max()
        plt.imshow(first_x[i].reshape(32, 32, 3))
        zca_i = zca_x[i]
        # -- the range of zca output is kind of arbitrary,
        #    it is unit-normal-ish
        mi = zca_i.min()
        ma = zca_i.max()
        print 'range', i, mi, ma
        assert mi > -3
        assert ma < 3
        plt.subplot(4, 2, 2 * i + 2)
        plt.imshow((zca_i.reshape(32, 32, 3) - mi) / (ma - mi + 1e-7))
    if SHOW:
        plt.show()




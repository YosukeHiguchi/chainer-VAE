import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)

    def __call__(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)

        return mu, ln_var

class Decoder(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

    def __call__(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)

        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2
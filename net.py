import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


class VAE(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)

            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(h_h, h_in)

    def __call__(self, x, sigmoid=True):
        z = self.encode(x)[0]
        return self.decode(z, sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)

        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)

        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

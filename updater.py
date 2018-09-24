import numpy as np

import chainer
from chainer import cuda
from chainer import Variable
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L


class VAEUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self.enc, self.dec = kwargs.pop('models')
        self._optimizers = kwargs.pop('optimizers')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        x = Variable(self.converter(batch, self.device))
        xp = cuda.get_array_module(x.data)

        enc = self.enc
        opt_enc = self._optimizers['enc']
        dec = self.dec
        opt_dec = self._optimizers['dec']

        mu, ln_var = enc(x)

        batchsize = len(mu.data)
        rec_loss = 0
        k = 10
        for l in range(k):
            z = F.gaussian(mu, ln_var)
            rec_loss += F.bernoulli_nll(x, dec(z, sigmoid=False)) / (k * batchsize)

        loss = rec_loss + 1.0 * F.gaussian_kl_divergence(mu, ln_var) / batchsize

        enc.cleargrads()
        dec.cleargrads()
        loss.backward()
        opt_enc.update()
        opt_dec.update()

        chainer.report({'rec_loss': rec_loss})
        chainer.report({'loss': loss})

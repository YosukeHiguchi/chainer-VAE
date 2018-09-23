import chainer
from chainer.dataset import convert

class VAEUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self._iterators = kwargs.pop('iterators')
        self.vae = kwargs.pop('models')
        self._optimizers = kwargs.pop('optimizers')
        self.device = kwargs.pop('device')

        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self);
        
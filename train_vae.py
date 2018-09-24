import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
from chainer import training
from chainer import cuda
from chainer.training import extensions

import net
from updater import VAEUpdater


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    parser.add_argument('--out', '-o', type=str, default='model',
                        help='path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(args)


    enc = net.Encoder(784, args.dimz, 500)
    dec = net.Decoder(784, args.dimz, 500)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        enc.to_gpu()
        dec.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    opt_enc = chainer.optimizers.Adam()
    opt_dec = chainer.optimizers.Adam()
    opt_enc.setup(enc)
    opt_dec.setup(dec)

    train, _ = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)

    updater = VAEUpdater(
        models=(enc, dec),
        iterators={
            'main': train_iter
        },
        optimizers={
            'enc': opt_enc,
            'dec': opt_dec
        },
        device=args.gpu,
        params={
        })
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.dump_graph('loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    log_keys = ['epoch', 'loss', 'rec_loss']
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(log_keys))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    main()
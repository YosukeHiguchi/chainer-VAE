import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from net import Decoder


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST')
    parser.add_argument'--enc_path', type=str
                        help='path to the encoder model')
    parser.add_argument('--dec_path', type=str
                        help='path to the decoder model')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.ou)

    print(args)

    enc = net.Encoder(784, args.dimz, 500)
    dec = net.Decoder(784, args.dimz, 500)
    chainer.serializers.load_npz(args.enc_path, enc)
    print('Encoder loaded successfully from {}'.format(args.enc_path))
    chainer.serializers.load_npz(args.dec_path, dec)
    print('Decoder loaded successfully from {}'.format(args.dec_path))

    _, test = chainer.datasets.get_mnist(withlabel=False)

    import pdb; pdb.set_trace()

    # def save_images(x, filename):
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    #     for ai, xi in zip(ax.flatten(), x):
    #         ai.imshow(xi.reshape(28, 28))
    #     fig.savefig(filename)
    #
    # enc.to_cpu()
    # dec.to_cpu()
    #
    # train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    # x = chainer.Variable(np.asarray(train[train_ind]))
    # with chainer.using_config('train', False):
    #     mu, _ = enc(x)
    #     x_gen = dec(mu)
    # save_images(x.data, os.path.join(args.out, 'train'))
    # save_images(x_gen.data, os.path.join(args.out, 'train_reconstructed'))
    #
    # z = chainer.Variable(np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
    # x_gen = dec(z)
    # save_images(x_gen.data, os.path.join(args.out, 'sampled'))

if __name__ == '__main__':
    main()
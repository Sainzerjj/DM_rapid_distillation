from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    gpus = [1]
    conf = cifar10_autoenc()
    train(conf, gpus=gpus)
#!/usr/bin/env python3
import os, urllib.request
import mxnet as mx
from os.path import expanduser
import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


PATH = 'http://data.mxnet.io/models/imagenet-11k/'

def download(url, prefix=''):
    filename = os.path.join(
        [
            expanduser("~"),
            '.imagetagging',
            prefix + url.split('/')[-1]
        ]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

def get_models():
    logging.info('Downloading')
    download(PATH+'resnet-152/resnet-152-symbol.json', 'full-')
    download(PATH+'resnet-152/resnet-152-0000.params', 'full-')
    filename = download(PATH+'synset.txt', 'full-')
    logging.info('Ready downloading')
    with open(filename):
    synset [l.rstrip() for l in f]
    return synset




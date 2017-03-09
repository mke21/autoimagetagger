#!/usr/bin/env python3
# coding: utf-8

import os, urllib.request
from os.path import expanduser
import subprocess

import mxnet as mx

import cv2
import matplotlib
from collections import namedtuple
import numpy as np

import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'

logging.basicConfig(level=logging.DEBUG)
Batch = namedtuple("Batch", ['data'])

PATH = 'http://data.mxnet.io/models/imagenet-11k/'

def download(url, prefix=''):
    filename = os.path.join(
        *[
            expanduser("~"),
            '.imagetagging',
            prefix + url.split('/')[-1]
        ]
    )
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

def get_models():
    logging.info('Downloading')
    download(PATH+'resnet-152/resnet-152-symbol.json', 'full-')
    download(PATH+'resnet-152/resnet-152-0000.params', 'full-')
    filename = download(PATH+'synset.txt', 'full-')
    logging.info('Ready downloading')
    with open(filename) as f:
        synset = [l.rstrip().split()[1].rstrip(', ') for l in f]
    return synset



def iterimages(homedir):
    for root, subFolders, files in os.walk(homedir):
        for f in files:
            if f.endswith('.jpg'):
                yield(os.path.join(root, f))


def prepaire():
    synsets = get_models()
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        os.path.join(
            *[
                expanduser("~"),
                '.imagetagging',
                'full-resnet-152'
            ]
        ), 0
    )
    mod = mx.mod.Module(
        symbol = sym,
        context = mx.cpu()
    )
    mod.bind(
        for_training=False,
        data_shapes =[('data', (1,3,224,224))]
    )
    mod.set_params(arg_params, aux_params)
    return {'mod': mod, 'synsets': synsets}

def predict(filename, mod, synsets, cutoff=0.1):
    """
    predicts the tags for an imagenet
    """
    logging.info('prediction %s', filename)
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    img = img[np.newaxis, :] 
    
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)

    a = np.argsort(prob)[::-1]
    return {
        'filename': filename,
        'tags': [synsets[i] for i in a[0:5] if prob[i] > cutoff]
        }

def savetag(filename, tags):
    """
    will add tags to tmsu
    """
    if tags:
        logging.info("Writing tags %s", filename)
        subprocess.call(
            [
                'tmsu',
                'tag',
                filename,
            ] + tags
        )
            


if __name__ == '__main__':
    import sys
    models = prepaire()
    for i in iterimages(sys.argv[1]):
        savetag(**predict(i, **models))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import word2vec
from utils.score_model import score_model
import sys

def main():
    '''
    System arguments:
    1 = Filename of model
    2 = Distance metric: e = euclidean; c = cosine
    '''

    fname = sys.argv[1]

    if sys.argv[2] == 'c':
        dist = 'cosine'
    elif sys.argv[2] == 'e':
        dist = 'euclidean'

    m = word2vec.Model('', dist_metric=dist, fname=fname, binary=False)
    acc = score_model(m)


if __name__ == '__main__':
    main()

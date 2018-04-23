#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.score_model import score_elmo_model
import sys

def main():
    '''
    System arguments:
    1 = # of dimensions for GloVe model
    2 = Distance metric: e = euclidean; c = cosine
    '''

    print("Running elmo model with d=", d_in, 'Dist metric=', dist)
    m = word2vec.Model('glove', dist_metric=dist, d=d_in)
    score_model(m)


if __name__ == '__main__':
    # word2vec google Total accuracy: 119/365 == 0.32602739726027397
    # d=50 Total accuracy: 119/367 == 0.3242506811989101
    main()

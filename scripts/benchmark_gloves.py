#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import word2vec
from utils.score_model import score_model
import sys

def main():
    '''
    System arguments:
    1 = # of dimensions for GloVe model
    2 = Distance metric: e = euclidean; c = cosine  
    '''
    
    if len(sys.argv) == 3:
        d_in = sys.argv[1]
        if sys.argv[2] == 'c':
            dist = 'cosine'
        elif sys.argv[2] == 'e':
            dist = 'euclidean'
    else:
        d_in = 50
        dist = 'euclidean'
        
    print("Running GloVe model with d=", d_in, 'Dist metric=', dist)
    m = word2vec.Model('glove', dist_metric=dist, d=d_in)
    score_model(m)


if __name__ == '__main__':
    # word2vec google Total accuracy: 119/365 == 0.32602739726027397
    # d=50 Total accuracy: 119/367 == 0.3242506811989101
    main()

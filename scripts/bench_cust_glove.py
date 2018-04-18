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

    dist = 'cosine'

    print('Testing acc on custom glove vectors')

    m = word2vec.Model('custglove', dist_metric='cosine', 
        d=300, fname='custom_glove_embed_w2v.txt', 
        data_dir='/root/w266_final/data', binary=False)

    score_model(m)


if __name__ == '__main__':
    # Total accuracy: /365 == 
    # Total accuracy: /367 == 
    main()

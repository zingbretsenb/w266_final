#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import word2vec
from utils.score_model import score_model
import sys

def main():
    '''
    '''

    dist = 'cosine'

    print('Testing acc on custom word2vec vectors')

    m = word2vec.Model('custw2v', dist_metric='cosine', 
        d=300, fname='short_w2v.txt', binary=False)

    score_model(m)


if __name__ == '__main__':
    # Total accuracy: /365 == 
    # Total accuracy: /367 == 
    main()

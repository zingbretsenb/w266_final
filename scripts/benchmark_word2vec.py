#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import word2vec
from utils.score_model import score_model


def main():
    m = word2vec.Model('word2vec', dist_metric='cosine', d=50)
    score_model(m)


if __name__ == '__main__':
    # word2vec google Total accuracy: 119/365 == 0.32602739726027397
    # d=50 Total accuracy: 119/367 == 0.3242506811989101
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import word2vec
from utils import data
import numpy as np


def score_correct(letter, index):
    a = ord('a')
    if ord(letter) == a + index:
        return 1
    else:
        return 0


def score_model(m):
    sat = data.FileFinder().get_sat_data()
    n_correct = n_total = 0

    for question in sat:
        try:
            q = question['question']
            # Only take the words, not the POS
            a = [ans[0] for ans in question['answers']]
            dists = m.score_answers(q, a)
            order = np.argsort(dists)
            sorted_dists = np.array(dists)[order]
            sorted_a = np.array(a)[order]

            print("------------")
            print("Question: {}".format(q))

            print("Sorted distances:")
            for dist, ans in zip(sorted_dists, sorted_a):
                print("Words: {}, score: {}".format(ans, dist))

            if m.dist_metric == 'euclidean':
                best = np.argmin(dists)
            elif m.dist_metric == 'cosine':
                best = np.argmax(dists)
            print('Best answer found: {}'.format(','.join(a[best])))
            print('Correct answer: {}'.format(question['correct'][0]))

            correct_letter = question['correct_letter']
            if score_correct(correct_letter, best):
                n_correct += 1
                print('Correct!')
            else:
                print('Incorrect :-(')

            n_total += 1
        except:
            q = question['question']

            print("------------")
            print("Question: {}".format(q))
            for ans in question['answers']:
                print("Words: {}".format(ans))
            print("Unknown words!")
            n_total += 1

    print("Total accuracy: {}/{} == {}".format(n_correct, n_total,
                                               n_correct/n_total))

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


def main():
    m = word2vec.Model('word2vec', dist_metric='euclidean')
    sat = data.read_sat_data()
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

            best = np.argmin(dists)
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
            pass

    print("Total accuracy: {}/{} == {}".format(n_correct, n_total,
                                               n_correct/n_total))


if __name__ == '__main__':
    # Total accuracy: 119/365 == 0.32602739726027397
    main()
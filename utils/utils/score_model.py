#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.spatial.distance import cosine
from utils import word2vec
from utils import data
from allennlp.commands.elmo import ElmoEmbedder


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
        except Exception as e:
            print(e)
            q = question['question']

            print("------------")
            print("Question: {}".format(q))
            for ans in question['answers']:
                print("Words: {}".format(ans))
            print("Unknown words!")
            n_total += 1

    print("Total accuracy: {}/{} == {}".format(n_correct, n_total,
                                               n_correct/n_total))

def score_elmo_model(style="pairs", toplayers=3, chooselayer=None):
    """
    Use all layers from an elmo model for scoring


    """

    assert toplayers in (1, 2, 3)
    assert style in ('single', 'pairs', 'dictionary')

    ee = ElmoEmbedder()

    sat = data.FileFinder().get_sat_data()
    n_correct = n_total = 0

    for question in sat:
        try:
            q = question['question']
            a = [ans[0] for ans in question['answers']]
            n_answers = len(a)

            if style == "pairs":
                q_embed = ee.embed_sentence(q)
                q_layers = q_embed[:,0] - q_embed[:,1]

                # Only take the words, not the POS
                a_layers = np.array([e[:,0] - e[:,1] for e in ee.embed_sentences(a)])

            elif style == "single":
                q_embed = np.array(list(ee.embed_sentences([[w] for w in q])))
                q_embed = np.array(q_embed).reshape(2,3,1024).transpose((1,0,2))
                q_layers = q_embed[:,0] - q_embed[:,1]


                word1 = np.array(list(ee.embed_sentences([[w[0]] for w in a])))
                word2 = np.array(list(ee.embed_sentences([[w[1]] for w in a])))
                a_embed = word1 - word2
                a_layers = a_embed.reshape((5, 3, 1024))

            # So that the first dimensions in both q and a is layers
            a_layers = a_layers.transpose(1, 0, 2)

            # If we just want one layer
            if chooselayer in (0, 1, 2):
                q_layers = q_layers[chooselayer].reshape(1, 1024)
                a_layers = a_layers[chooselayer].reshape(1, -1, 1024)

            # Take top N layers
            else:
                q_layers = q_layers[-toplayers:]
                a_layers = a_layers[-toplayers:]

            dists = []
            for i, (ql, als) in enumerate(zip(q_layers, a_layers)):
                for al in als:
                    dists.append(cosine(ql, al))

            order = np.argsort(dists)
            sorted_dists = np.array(dists)[order]
            sorted_a = np.tile(np.array(a), (3, 1))[order]

            print("------------")
            print("Question: {}".format(q))

            print("Sorted distances:")
            for dist, ans in zip(sorted_dists, sorted_a):
                print("Words: {}, score: {}".format(ans, dist))

            best = np.argmin(dists) % n_answers

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

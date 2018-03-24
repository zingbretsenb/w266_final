#!/usr/bin/env python

import gensim
from . import data
import math
import numpy as np



def get_euclidean_dist(a, b):
    return math.sqrt(np.square(a - b).sum())


class Model:
    def __init__(self, model, dist_metric, d=50):
        """Logic for storing vectors and scoring analogies"""
        self.model = model.lower()
        self.dist_metric = dist_metric.lower()

        if model == "word2vec":
            try:
                data_file = data.WORD2VEC_FILE
                self.vectors = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                                             binary=True)
            except:
                print('word2vec vectors available here:')
                print('https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit')
        elif model == 'glove':
            self.d = str(d)
            data_file = data.GLOVE_WORD2VEC_FILE.format(self.d)
            print("Loading {}".format(data_file))
            try:
                self.vectors = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                                               binary=False)
            except:
                print('Could not load {}'.format(data_file))
                print('Maybe try a different embedding dimension?')

        if self.dist_metric == 'euclidean':
            self.get_dist = get_euclidean_dist


    def get_word2vec_embedding(self, word):
        return self.vectors.word_vec(word)


    def get_embedding(self, word):
        return self.get_word2vec_embedding(word)


    def get_difference_vector(self, w1, w2):
        e1 = self.get_embedding(w1)
        e2 = self.get_embedding(w2)
        return e2 - e1


    def score_answers(self, question, answers):
        """Given pairs of question and answer words, return the nearest neighbors"""
        ques_embed = self.get_difference_vector(*question)
        ans_embed = [self.get_difference_vector(*a) for a in answers]

        dists = [self.get_dist(ques_embed, a) for a in ans_embed]
        return dists


if __name__ == '__main__':
    m = Model(model='word2vec', dist_metric='euclidean')
    question = ['king', 'queen']
    answers = [['man', 'woman'], ['boy', 'girl'], ['car', 'icicle']]
    dists = m.score_answers(question, answers)

#!/usr/bin/env python

import gensim
from . import data
import math
import numpy as np
from collections import defaultdict
import os



def get_euclidean_dist(a, b):
    return math.sqrt(np.square(a - b).sum())

def get_cos_dist(a, b):
    top = np.inner(a,b)
    bottom = np.dot(np.linalg.norm(a), np.linalg.norm(b))
    result = np.divide(top, bottom)
    return result

class Model:
    def __init__(self, model, dist_metric, data_dir=None, d=50, fname=None, binary=None):
        """Logic for storing vectors and scoring analogies"""
        self.model = model.lower()
        self.dist_metric = self.set_dist_metric(dist_metric.lower())
        self.source_correct = defaultdict(lambda: 0)
        self.pos_correct = defaultdict(lambda: 0)
        self.source_total = defaultdict(lambda: 0)
        self.pos_total = defaultdict(lambda: 0)

        self.set_dist_metric(dist_metric)

        if data_dir is not None:
            self.finder = data.FileFinder(data_dir)
        else:
            self.finder = data.FileFinder()

        if model == "word2vec":
            try:
                data_file = self.finder.get_file('WORD2VEC_FILE')
                self.vectors = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                                             binary=True)
            except:
                print('word2vec vectors available here:')
                print('https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit')

        elif model == 'glove':
            self.d = str(d)
            data_file = self.finder.get_file('GLOVE_WORD2VEC_FILE').format(self.d)
            print("Loading {}".format(data_file))
            try:
                self.vectors = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                                               binary=False)
            except:
                print('Could not load {}'.format(data_file))
                print('Maybe try a different embedding dimension?')

        elif fname is not None and binary is not None:
            try:
                data_file = os.path.join(self.finder.data_dir, fname)
                self.vectors = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                                               binary=binary)
            except:
                print('Was not able to load: {}',format(fname))
                print('Binary was set to: {}',format(binary))

        self.data_file = data_file


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return """Model: {}\nFile: {}\nDistance: {}""".format(
            self.model, self.data_file, self.dist_metric)


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


    def set_dist_metric(self, metric):
        self.dist_metric = metric
        if metric == 'euclidean':
            self.get_dist = get_euclidean_dist
        elif metric == 'cosine':
            self.get_dist = get_cos_dist


if __name__ == '__main__':
    m = Model(model='word2vec', dist_metric='euclidean')
    question = ['king', 'queen']
    answers = [['man', 'woman'], ['boy', 'girl'], ['car', 'icicle']]
    dists = m.score_answers(question, answers)

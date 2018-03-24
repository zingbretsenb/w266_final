#!/usr/bin/env python

import gensim
import data

data_file = data.WORD2VEC_FILE
model = gensim.models.KeyedVectors.load_word2vec_format(data_file,
                                                        binary=True)


def get_embedding(word, model=model):
    return model.wv.word_vec(word)


if __name__ == '__main__':
    woman = model.wv.word_vec('woman')

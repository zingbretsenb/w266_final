#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.scripts.glove2word2vec import glove2word2vec
from utils import data


def main():
    """Converts glove txt files to word2vec format"""
    for dim in (50, 100, 200, 300):
        data_file = data.GLOVE_TXT_FILE.format(dim)
        output_file = data.GLOVE_WORD2VEC_FILE.format(dim)
        print("Converting {} to {}".format(data_file, output_file))
        glove2word2vec(data_file, output_file)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from gensim.scripts.glove2word2vec import glove2word2vec
from utils import data


def main():
    """Converts glove txt files to word2vec format"""
    for dim in (50, 100, 200, 300):
        data_file = data.FileFinder().get_file('GLOVE_TXT_FILE').format(dim)
        output_file = data.FileFinder().get_file('GLOVE_WORD2VEC_FILE').format(dim)
        print("Converting {} to {}".format(data_file, output_file))
        glove2word2vec(data_file, output_file)


def convert_large_gloves():
    finder = data.FileFinder()
    data_dir = finder.data_dir
    for data_file in ('glove.42B.300d.txt', 'glove.twitter.27B.200d.txt'):
        output_file = data_file.split('.')
        output_file[-1] = 'word2vec'
        output_file = '.'.join(output_file)
        data_file = os.path.join(data_dir, data_file)
        output_file = os.path.join(data_dir, output_file)
        print("Converting {} to {}".format(data_file, output_file))
        glove2word2vec(data_file, output_file)

if __name__ == '__main__':
    convert_large_gloves()

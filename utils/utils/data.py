#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from os.path import expanduser, join
from configparser import ConfigParser

home = expanduser("~")
config = ConfigParser()
config.read_file(open(join(home, ".w266_config"), 'r'))
data_dir = config.get('w266', 'data_dir')

class FileFinder:
    def __init__(self, data_dir=data_dir):
        self.last_file = None
        self.data_dir = data_dir
        self.files = {
            'WORD2VEC_FILE': os.path.join(data_dir, "GoogleNews-vectors-negative300.bin"),
            'GLOVE_TXT_FILE': os.path.join(data_dir, "glove.6B.{}d.txt"),
            'GLOVE_WORD2VEC_FILE': os.path.join(data_dir, "glove.6B.{}d.word2vec"),
            'GLOVE42B_WORD2VEC_FILE': os.path.join(data_dir, "glove.42B.300d.word2vec"),
            'GLOVE42B_TXT_FILE': os.path.join(data_dir, "glove.42B.300d.txt"),
            'GLOVE840B_WORD2VEC_FILE': os.path.join(data_dir, "glove.840B.300d.word2vec"),
            'GLOVE840B_TXT_FILE': os.path.join(data_dir, "glove.840B.300d.txt"),
            'RAW_SAT_DATA_FILE': os.path.join(data_dir, "SAT-package-V3.txt"),
            'JSON_SAT_DATA_FILE': os.path.join(data_dir, "SAT.json")
        }


    def show_filenames(self):
        for k, v in self.files.items():
            print(k, v)


    def get_file(self, fname):
        self.last_file = fname
        return self.files[fname]


    def get_last_filename(self):
        return self.last_file


    def get_sat_data(self):
        """Loads sat file as a generator"""
        fpath = self.get_file('JSON_SAT_DATA_FILE')
        with open(fpath, 'r') as f:
            lines = f.readlines()

        return (json.loads(line) for line in lines)


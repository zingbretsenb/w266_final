#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

package_directory = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(package_directory, '../../data')
WORD2VEC_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
GLOVE_TXT_FILE = os.path.join(DATA_DIR, "glove.6B.{}d.txt")
GLOVE_WORD2VEC_FILE = os.path.join(DATA_DIR, "glove.6B.{}d.word2vec")
RAW_SAT_DATA_FILE = os.path.join(DATA_DIR, "SAT-package-V3.txt")
JSON_SAT_DATA_FILE = os.path.join(DATA_DIR, "SAT.json")


def read_sat_data():
    """Loads sat file as a generator"""
    with open(JSON_SAT_DATA_FILE, 'r') as f:
        lines = f.readlines()
    return (json.loads(line) for line in lines)

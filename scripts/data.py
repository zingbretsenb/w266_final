#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

package_directory = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(package_directory, '../data')
RAW_SAT_DATA_FILE = os.path.join(DATA_DIR, "SAT-package-V3.txt")
JSON_SAT_DATA_FILE = os.path.join(DATA_DIR, "SAT.json")


def read_sat_data():
    """Loads sat file as a generator"""
    with open(JSON_SAT_DATA_FILE, 'r') as f:
        lines = f.readlines()
    return (json.loads(line) for line in lines)

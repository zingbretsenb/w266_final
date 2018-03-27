#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import data

finder = data.FileFinder()
sat = finder.get_sat_data()

with open('pos_and_src.txt', 'w') as f:
    f.write('POS1,POS2,Source')
    for q in sat:
        f.write('\n' + ','.join(q['question_POS'] + q['source'].split()[:1]))

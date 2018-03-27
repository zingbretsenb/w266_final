#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import data

finder = data.FileFinder()
sat = finder.get_sat_data()

with open('pos.txt', 'w') as f:
    f.write('POS1,POS2')
    for q in sat:
        f.write('\n' + ','.join(q['question_POS']))

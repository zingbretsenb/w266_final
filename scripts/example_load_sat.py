#!/usr/bin/env python
# -*- coding: utf-8 -*-

# From inside the scripts folder you can `import data` directly
# From the root of the repository, you'll have to do:
# `from scripts import data`
import data

questions = data.read_sat_data()

for q in questions:
    # Do things
    print(q['question'])
    print(q['answers'])
    print("")

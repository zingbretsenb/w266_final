#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

fname = "data/SAT-package-V3.txt"
output = 'data/SAT.json'

def split_answer(line, delimiter=" "):
    split_line = line.split(delimiter)
    return split_line[0:2], split_line[2:]


with open(fname, 'r') as f:
    lines = f.readlines()

lines = [line for line in lines if line[0] != "#"]
questions = []
question = {}
answers = []
new_ques = False
ques = False
answer = 0
correct = False

for line in lines:
    line = line.strip()

    if line == '':
        if len(question.keys()) > 0:
            correct_letter = answers.pop()
            question['correct'] = answers[ord(correct_letter) - ord('a')]
            question['answers'] = [split_answer(a) for a in answers]
            questions.append(json.dumps(question))
        question = {}
        new_ques = True
        continue

    if new_ques:
        question['source'] = line
        new_ques = False
        ques = True
        answers = []
        continue

    if ques:
        question['question'], question['question_POS'] = split_answer(line)
        ques = False
        answer = 0
        continue


    answers.append(line)
    sp = line.split()
    if len(sp) > 3:
        print(line)


    # try:
    #     question['correct'] = answers[ord(line) - 97]
    # except:
    #     print(line)
    #     raise

with open(output, 'w') as f:
    for q in questions:
        f.write(q + '\n')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from utils import data


if __name__ == '__main__':


    def split_answer(line, delimiter=" "):
        """Split each line in the [word1, word2], [POS1:POS2]"""
        split_line = line.split(delimiter)
        words, pos = split_line[0:2], split_line[2:]
        return words, pos[0].split(":")

    fname = data.RAW_SAT_DATA_FILE
    output = data.JSON_SAT_DATA_FILE

    with open(fname, 'r') as f:
        lines = f.readlines()

    # Filter out lines from header
    lines = [line for line in lines if line[0] != "#"]

    questions = answers = []
    question = {}
    new_ques = ques = correct = False
    answer = 0

    for line in lines:
        line = line.strip()

        # A blank line separates each question
        # On each change of question, add the previous question to the array
        if line == '':
            if len(question.keys()) > 0:
                # The last item in the answers is the letter of the correct answer
                # Remove that from the answers array
                correct_letter = answers.pop()
                # Use that letter to index into the answers array
                question['correct_letter'] = correct_letter
                question['correct'] = split_answer(answers[ord(correct_letter) - ord('a')])
                question['answers'] = [split_answer(a) for a in answers]
                questions.append(json.dumps(question))

            # Reset the question dictionary and get ready to read in source
            question = {}
            new_ques = True
            continue

        # Read in source of question and get ready to read in question
        if new_ques:
            question['source'] = line
            new_ques = False
            ques = True
            answers = []
            continue

        # Read in question and get ready to read in answers
        if ques:
            question['question'], question['question_POS'] = split_answer(line)
            ques = False
            answer = 0
            continue

        # Read in the remaining lines as answers
        # (including the correct letter)
        # (this will be removed from the list later)
        answers.append(line)
        sp = line.split()
        if len(sp) > 3:
            print(line)

    with open(output, 'w') as f:
        for q in questions:
            f.write(q + '\n')

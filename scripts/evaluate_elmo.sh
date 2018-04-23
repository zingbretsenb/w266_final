#!/bin/bash

python evaluate_sem_syn.py --vectors_file elmo0.txt > results/elmo0_syn_sem.txt &
python evaluate_sem_syn.py --vectors_file elmo1.txt > results/elmo1_syn_sem.txt &
python evaluate_sem_syn.py --vectors_file elmo2.txt > results/elmo2_syn_sem.txt &


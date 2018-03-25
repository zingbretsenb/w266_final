#!/bin/bash

python benchmark_word2vec.py e > w2v_e.txt
python benchmark_word2vec.py c > w2v_c.txt
python benchmark_gloves.py 50 e > glove50_e.txt
python benchmark_gloves.py 50 c > glove50_c.txt
python benchmark_gloves.py 100 e > glove100_e.txt
python benchmark_gloves.py 100 c > glove100_c.txt
python benchmark_gloves.py 200 e > glove200_e.txt
python benchmark_gloves.py 200 c > glove200_c.txt
python benchmark_gloves.py 300 e > glove300_e.txt
python benchmark_gloves.py 300 c > glove300_c.txt

echo 'COMPLETE!'

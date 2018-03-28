#!/usr/bin/env bash

for B in 840 42;
do
	for dist in e c
	do python benchmark_nonbinary.py glove.${B}B.300d.word2vec $dist > glove${B}b_300_${dist}.txt
	done
done

for dist in e c
do
	python benchmark_nonbinary.py glove.twitter.27B.200d.word2vec $dist > glove27B_twitter_200_${dist}.txt
done

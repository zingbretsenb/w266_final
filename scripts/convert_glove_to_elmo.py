import os
import h5py

from tqdm import tqdm
from gensim.scripts.glove2word2vec import glove2word2vec
from utils import data, word2vec
from allennlp.commands.elmo import ElmoEmbedder


def convert_glove_to_w2v():
    infile = '/root/cove/test/.vector_cache/glove.6B.100d.txt'
    outfile = '/root/w266_final/data/glove.6B.100d.txt'
    glove2word2vec(infile, outfile)
    return word2vec.Model('glove6B', 'cosine', d=100, fname='glove.6B.100d.txt', binary=False)


def convert_vecs_to_elmo(m):
    ee = ElmoEmbedder()
    elmos = {}
    for word in tqdm(m.vectors.vocab):
        elmos[word] = ee.embed_sentence([word])
    return elmos


def main():
    m = convert_glove_to_w2v()
    elmo_vectors = convert_vecs_to_elmo(m)
    with h5py.File('elmos.vecs', 'w') as f:
        # f.create_dataset('elmo', data=elmo_vectors)
        f.create_dataset('elmo', data=elmos)


if __name__ == '__main__':
    main()

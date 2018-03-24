# w266_final
Word embeddings for analogies

Originally set up on OS X with Python Version: 
sys.version_info(major=3, minor=6, micro=4, releaselevel='final', serial=0)

`setup.sh` should set up the virtual env, and helper scripts are installed in the `utils` package.

`pip install -e <package>` installs the package into the environment, but links to the source rather than moving the files, so changes to files in the `utils/utils/` directory will be reflected in python without having to reinstall the package.

Current word embedding benchmarks on analogy dataset:
word2vec (GoogleNews-vectors-negative300.bin): 119/365 == 0.32602739726027397
glove.6B.50d: 119/367 == 0.3242506811989101
glove.6B.300d: 127/367 == 0.3460490463215259

# w266_final
Word embeddings for analogies

Originally set up on OS X with Python Version: 
sys.version_info(major=3, minor=6, micro=4, releaselevel='final', serial=0)

`setup.sh` should set up the virtual env, and helper scripts are installed in the `utils` package.

`pip install -e <package>` installs the package into the environment, but links to the source rather than moving the files, so changes to files in the `utils/utils/` directory will be reflected in python without having to reinstall the package.

If you want to use `ELMo`, you will also need to `pip install allennlp`

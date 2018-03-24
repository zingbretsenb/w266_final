#!/usr/bin/env bash

# Sets up a virtual environment and installs the required packages
python -m venv ./env
source env/bin/activate
pip install -r requirements.txt
pip install -e ./utils

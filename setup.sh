#!/bin/bash

# create virtual environment
python3.9 -m venv env

# activate virtual environment 
source ./env/bin/activate

# install reqs 
echo -e "[INFO:] Installing necessary requirements..."
python3.9 -m pip install -r requirements.txt

# deactivate env 
deactivate

# happy user msg 
echo -e "[INFO:] Setup complete!"
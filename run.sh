#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run simple image search
echo -e "[INFO:] Running SIMPLE image search on default image ..." # user msg 
python3 src/image_search.py -alg SIMPLE

# run KNN image search
echo -e "[INFO:] Running KNN image search on default image ..." # user msg 
python3 src/image_search.py -alg KNN

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "[INFO:] Image searches complete!"
#!bin/bash

conda install --file ./conda.txt

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

# enable extenstions
jupyter nbextensions_configurator enable --user

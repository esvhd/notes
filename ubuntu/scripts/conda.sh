#!bin/bash

conda install --file ./conda.txt

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

# enable extenstions
jupyter nbextensions_configurator enable --user

# a few packages can be installed from github source in editable mode
# filterpy
# bastage/arch
# fastai

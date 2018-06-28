#!/bin/bash

conda env create -f ./tf_conda_env.yml

exit_stats = $?
if [ $exit_stats -eq 0 ]; then
    conda activate tf
    if [ $? -eq 0 ]; then
        echo Link environment to Jupyter notebook...
        python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
    else
        echo Not in new environment!
    fi
fi

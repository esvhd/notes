#!/bin/bash

# As of 2018-10-03
# need cuda 9.2 at least for gcc 7
# for mult-gpu enable NCCL

# R GPU support
cmake .. -DUSE_CUDA=ON -DR_LIB=ON -DUSE_NCCL=ON -DNCCL_ROOT=/usr/local/nccl
make install -j

# start R, run below
# install.packages('~/git/xgboost/build/R-package', repos=NULL, type='source')

# python
# cmake .. -DUSE_CUDA=ON  -DUSE_NCCL=ON -DNCCL_ROOT=/usr/local/nccl
# make -j

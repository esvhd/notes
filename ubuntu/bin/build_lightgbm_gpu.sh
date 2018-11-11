#!/bin/bash

# install the libs required
# sudo apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev
# clinfo to show platform / GPU IDs, should be in ubuntu notes too.
# sudo apt-get install clinfo

# git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
# mkdir build ; cd build

# cmake -DUSE_GPU=1 ..
# if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j8

# install python api
cd ../python-package/
python setup.py install --precompile

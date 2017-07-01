#!/bin/bash

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zwl/lib/blpapi_cpp/Linux

export LD_LIBRARY_PATH

export BLPAPI_ROOT=/home/zwl/lib/blpapi_cpp

export FRED_API_KEY=4ab8daf548f583dcfa16a95383c1adbd

PYTHONPATH=/home/zwl/git/pythonlib/:/home/zwl/git/pyhistdata:/home/zwl/git/pypbo:/home/zwl/git/drivetools
export PYTHONPATH

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1

/usr/anaconda3/bin/jupyter-notebook --no-browser --notebook-dir /home/zwl 1>/tmp/cron.jupyter 2> /tmp/cron.jupyter.err &

#!/bin/bash

sudo unlink /usr/local/cuda && sudo ln -s /usr/local/cuda-9.2 /usr/local/cuda
sudo unlink /usr/local/nccl && sudo ln -s /usr/local/nccl_2.3.5-2+cuda9.2 /usr/local/nccl


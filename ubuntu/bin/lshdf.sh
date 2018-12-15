#!/usr/anaconda3/bin/python
# List keys in a HDF5 file

import pandas as pd
import argparse
import os.path as osp
import sys

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, nargs=1, help='Input HDF5 file')

args = parser.parse_args()

print(args)

print(f'Reading from file {args.input_file[0]}')

if not osp.exists(args.input_file[0]):
    print('File does not exist. Exit.')
    sys.exit(1)

with pd.HDFStore(args.input_file[0]) as s:
    print(s.keys())


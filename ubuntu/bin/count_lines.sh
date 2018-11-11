#!/bin/bash

# pattern='*.py'
# echo "'$pattern'"

find . -name '*.py' | xargs wc -l

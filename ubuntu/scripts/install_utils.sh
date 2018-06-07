#!/bin/bash

umask 022

sudo apt-get -y install sysstat
sudo apt-get -y install curl
sudo apt-get -y install git

# ssh server
sudo apt-get -y install openssh-server
sudo service ssh status

# some packages needed for R packages
sudo apt-get -y install zlib1g-dev
sudo apt-get -y install libssl-dev
sudo apt-get -y install libcurl4-openssl-dev
sudo apt-get -y install r-base

sudo apt-get -y install gfortran-7
sudo apt-get -y install libgfortran-7-dev

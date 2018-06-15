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

sudo apt-get -y install gdebi

# apple time capsule
sudo apt-get -y install cifs-utils
sudo apt-get -y install smbclient

# chinese input, need reboot
# sudo apt-get install sogoupinyin

sudo apt-get -y install net-tools
#sudo apt-get -y install wireless-tools

#!/bin/bash

# Usage: histdata_get.sh directory_name


echo "Download directory $HISTDATA_FTP/$1"
echo "Pattern : DAT_ASCII_*$1.*"

wget -m --user=$HISTDATA_USER --password=$HISTDATA_PASS \
-A "DAT_ASCII_*$1.*" ftp://$HISTDATA_FTP/$1

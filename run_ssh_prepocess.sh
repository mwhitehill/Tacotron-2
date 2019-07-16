#!/bin/bash

echo Copying metadata
mkdir ~/Tacotron-2/data
cp /hdfs/intvc/t-mawhit/Tacotron-2/data/metadata_emt4.txt ~/Tacotron-2/data/metadata_emt4.txt
echo Copying raw audio data
mkdir ~/data
cp -r /hdfs/intvc/t-mawhit/data/Zo ~/data

echo running preprocess
python3 preprocess.py
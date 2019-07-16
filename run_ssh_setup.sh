#!/bin/bash

sudo apt-get -y update
sudo apt-get -y install portaudio19-dev
sudo apt-get -y install vim
sudo apt-get -y install libc6-dbg gdb valgrind

echo Set PYTHONPATH. new PYTHONPATH:
export PYTHONPATH=${PYTHONPATH}:${HOME}/Tacotron-2/code
python3 -c "import sys; print(sys.path)"


echo Installing pip requirements
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install numpy
sudo python3 -m pip install -r /hdfs/intvc/t-mawhit/Tacotron-2/code/requirements.txt
sudo python3 -m pip install tensorflow

mkdir ~/Tacotron-2
mkdir ~/Tacotron-2/code
echo Copying Code
cp -r /hdfs/intvc/t-mawhit/Tacotron-2/code/* ~/Tacotron-2/code

cd ~/Tacotron-2/code/
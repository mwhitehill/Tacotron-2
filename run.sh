#!/bin/bash

while getopts d:m:l:x option
do
    case "${option}"
    in
        d) DATA_DIR=${OPTARG};;
        l) LOG_DIR=${OPTARG};;
        m) MODEL_DIR=${OPTARG};;
    esac
done

echo DATA_DIR:
echo $DATA_DIR
echo MODEL_DIR:
echo $MODEL_DIR
echo PWD:
pwd
echo ls:
ls

mkdir ~/Tacotron-2
mkdir ~/Tacotron-2/code
cp -r $DATA_DIR/Tacotron-2/code/* ~/Tacotron-2/code
mkdir ~/Tacotron-2/data
cp -r $DATA_DIR/Tacotron-2/data/metadata_emt4.txt ~/Tacotron-2/data/metadata_emt4.txt
mkdir ~/data
cp -r $DATA_DIR/data/Zo ~/data

cd ~/Tacotron-2/code/

#pip3 install -r requirements.txt

if [ $OMPI_COMM_WORLD_RANK -eq 0 ]
then
	python --version
	python3 --version
    python3 preprocess.py --TEST
	mkdir $DATA_DIR/Tacotron-2/data/emt4_test
	cp ~/Tacotron-2/data/emt4_test $DATA_DIR/Tacotron-2/data/emt4_test
#	python preprocess.py --data_dir $DATA_DIR --save_dir $MODEL_DIR --nb_filters1 32 --nb_filters2 64 --nb_filters3 128 --dropout_rate1 0.25 --dropout_rate2 0.5 --nb_dense 256 --cv_split 0 --nb_task 2
else
    echo "Shut down"
fi
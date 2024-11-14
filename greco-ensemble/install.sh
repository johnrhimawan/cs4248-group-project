#!/bin/bash

#SBATCH --job-name=install_greco
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0851472@u.nus.edu

wget -P models https://sterling8.d2.comp.nus.edu.sg/~reza/GRECO/checkpoint.bin
wget https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz
tar -xf m2scorer.tar.gz

python -m venv .venv/
source .venv/bin/activate
pip install -r requirements.txt
pip install 'protobuf==3.20.0'

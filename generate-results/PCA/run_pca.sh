#!/bin/bash

data_dir="../../data/"

run_pca () {
    local desc_name=$1
    local set_name=$2
    local fname="$data_dir/dataset_${desc_name}_${set_name}.h5"
    python3 ./PCA_DATA.py $fname -v -if
}

run_pca $1 $2

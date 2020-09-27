#!/bin/bash

data_dir="../../data/"

run_fps () {
    local desc_name=$1
    local set_name=$2
    local step=$3
    local cols=$4
    local fname="$data_dir/dataset_${desc_name}_${set_name}.h5"
    if [ "$5" == "1" ]; then
        python3 ./FPS_DATA_NUMBA.py $fname -p $step -c $cols -nonan
    else
        python3 ./FPS_DATA_NUMBA.py $fname -p $step -c $cols
    fi
}

run_fps $1 $2 $3 $4 $5 

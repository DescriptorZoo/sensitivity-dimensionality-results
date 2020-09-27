#!/bin/bash

# RR and KRR on Si data

data_dir="../../data/"

test_train_split="train_test_split_list.txt"
targets="targets.txt"
krr_outputs="output-krr.txt"
rr_outputs="output-rr.txt"
krr_header="KRR_"
rr_header="RR_"

ACE_file="$data_dir/dataset_ACE_Si.h5"
SOAP_file="$data_dir/dataset_SOAP_Si.h5"
SOAPLITE_file="$data_dir/dataset_SOAPLITE_Si.h5"
CHSF_file="$data_dir/dataset_CHSF_Si.h5"
ACSF_file="$data_dir/dataset_ACSF_Si.h5"
ACSFX_file="$data_dir/dataset_ACSFX_Si.h5"
MBTR_file="$data_dir/dataset_MBTR_Si.h5"

ACE_cols=63
SOAP_cols=450
SOAPLITE_cols=450
CHSF_cols=20
ACSF_cols=51
ACSFX_cols=57
MBTR_cols=182

run_linreg () {
    local desc_name=$1
    local cols=$2
    local alpha1=$3
    local degree=$4
    local gamma=$5
    local alpha2=$6
    local f=1
    local s=1
    local fname="$data_dir/dataset_${desc_name}_Si.h5"
    local n=$cols
    for i in `seq 1 $n`; do
        python3 ./LINREG_DATA.py $fname $targets -s $test_train_split -g -lsq -krr -alpha $alpha1 -degree $degree -kernel rbf -nrm -targets1 -ref1 -gamma $gamma -x0 -f $f >> ${desc_name}${krr_outputs}
        python3 ./LINREG_DATA.py $fname $targets -s $test_train_split -g -qr -l2 -alpha $alpha2 -targets1 -ref1 -nrm -x0 -f $f >> ${desc_name}${rr_outputs}
        echo "$i `tail -n 1 ${desc_name}${krr_outputs}`" >> ${krr_header}${desc_name}_Si
        echo "$i `tail -n 1 ${desc_name}${rr_outputs}`" >> ${rr_header}${desc_name}_Si
        f=$(($f + $s))
    done
}

run_linreg $1 $2 $3 $4 $5 $6


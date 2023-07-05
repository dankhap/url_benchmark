#!/bin/bash
echo "in bash script"
export PATH=$PATH:/opt/conda/envs/urlb2/bin:/opt/conda/condabin
echo "all arguments: $@"
eval "$(conda shell.bash hook)"
conda activate urlb2
if [ $1 = "pretrain" ]; then
	python pretrain.py group_name=$2 agent=$3 domain=$4 seed=$5 obs_type=$6
elif [ $1 = "pre" ]; then
	python pretrain.py ${@:2}
elif [ $1 = "fine" ]; then
	python finetune.py ${@:2}
elif [ $1 = "finetune" ]; then
	python finetune.py group_name=$2 pretrained_agent=$3 task=$4 snapshot_ts=$5  seed=$6 obs_type=$7
elif [ $1 = "finetuned" ]; then
	python finetune_double.py group_name=$2 pretrained_agent=$3 task=$4 snapshot_ts=$5  seed=$6 obs_type=$7
fi


#!/bin/bash

algo=${1}
obs=${2}
env=${3}
task=${4}
seed=${5}

echo "algo=${algo}, obs=${obs}, env=${env}, task=${task}, seed=${seed}"
echo "starting fine..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_${obs}_finetune pretrained_agent=${algo} task=${env}_${task} snapshot_ts=2000000 seed=${seed} batch_sched=normal obs_type_params=${obs} " ./urlb_job_git.sh
echo "starting buffered..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_${obs}_buffered pretrained_agent=${algo} task=${env}_${task} snapshot_ts=0 seed=${seed} batch_sched=normal obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed}" ./urlb_job_git.sh
echo "starting buffered_partial..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_${obs}_buffered_partial pretrained_agent=${algo} task=${env}_${task} snapshot_ts=2000000 batch_sched=normal seed=${seed} obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed} load_only_encoder=true" ./urlb_job_git.sh


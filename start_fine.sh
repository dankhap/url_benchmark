#!/usr/bin/bash
algo=${1}
obs=${2}
env_task=${3}
env=${env_task%%'_'*}
seed=${4}
fine=${5}

echo "algo=${algo}, obs=${obs}, env_task=${env_task}, env=${env}, seed=${seed}"

if [[ $fine == "clean" || $fine == "*" ]]; then
	echo "starting clean..."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_clean pretrained_agent=${algo} task=${env_task} snapshot_ts=0 seed=${seed} batch_sched=normal obs_type_params=${obs} " ./urlb_job_git.sh
fi

if [[ $fine == "fine" || $fine == "*" ]]; then
	echo "starting fine..."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_finetune pretrained_agent=${algo} task=${env_task} snapshot_ts=2000000 seed=${seed} batch_sched=normal obs_type_params=${obs} " ./urlb_job_git.sh
fi

if [[ $fine == "partial" || $fine == "*" ]]; then

	echo "started finetune partial"
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_partial pretrained_agent=${algo} task=${env_task} snapshot_ts=2000000 seed=${seed} load_only_encoder=true batch_sched=normal obs_type_params=${obs} " ./urlb_job_git.sh
fi

if [[ $fine == "buffered" || $fine == "*" ]]; then
	echo "starting buffered..."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_buffered pretrained_agent=${algo} task=${env_task} snapshot_ts=0 seed=${seed} batch_sched=normal obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed}" ./urlb_job_git.sh
fi

if [[ $fine == "buffered_fast" || $fine == "*" ]]; then
	echo "starting buffered_fast..."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_buffered_fast pretrained_agent=${algo} task=${env_task} snapshot_ts=0 seed=${seed} batch_sched=fast obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed}" ./urlb_job_git.sh
fi

if [[ $fine == "buffered_partial" || $fine == "*" ]]; then
	echo "starting buffered_partial..."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_buffered_partial pretrained_agent=${algo} task=${env_task} snapshot_ts=2000000 batch_sched=normal seed=${seed} obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed} load_only_encoder=true" ./urlb_job_git.sh
fi

if [[ $fine == "buffered_partial_fast" || $fine == "*" ]]; then
	echo "starting buffered_partial fast.."
	sbatch --export=ALL,A="fine group_name=${algo}_${env_task}_${obs}_buffered_partial_fast pretrained_agent=${algo} task=${env_task} snapshot_ts=2000000 batch_sched=normal seed=${seed} obs_type_params=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed} load_only_encoder=true batch_sched=fast" ./urlb_job_git.sh
fi


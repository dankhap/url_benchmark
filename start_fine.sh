#!/bin/bash

algo=${1}
obs=${2}
env=${3}
task=${4}
seed=${5}

echo "algo=${algo}, obs=${obs}, env=${env}, task=${task}, seed=${seed}"
echo "starting fine..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_finetune pretrained_agent=${algo} task=${env}_${task} snapshot_ts=2000000 seed=${seed} batch_sched=normal obs_type=${obs} " ./urlb_job_git.sh
echo "starting buffered..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_buffered pretrained_agent=${algo} task=${env}_${task} snapshot_ts=0 seed=${seed} batch_sched=normal obs_type=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed}" ./urlb_job_git.sh
echo "starting buffered_partial..."
sbatch --export=ALL,A="fine group_name=${algo}_${env}_${task}_buffered_partial pretrained_agent=${algo} task=${env}_${task} snapshot_ts=2000000 batch_sched=normal seed=${seed} obs_type=${obs} buffer_dir=/code/url_benchmark/${algo}_${env}_${obs}_buffer${seed} load_only_encoder=true" ./urlb_job_git.sh

# sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_fix pretrained_agent=${1} task=walker_flip snapshot_ts=0 seed=16 obs_type=pixels action_repeat=2 batch_sched=normal buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_partial_fix pretrained_agent=${1} task=walker_flip snapshot_ts=2000000 batch_sched=normal seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_fix pretrained_agent=${1} task=walker_flip snapshot_ts=0 seed=20 obs_type=pixels action_repeat=2 batch_sched=normal buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_partial_fix pretrained_agent=${1} task=walker_flip snapshot_ts=2000000 seed=16 obs_type=pixels batch_sched=normal buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh

# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered pretrained_agent=${1} task=quadruped_run snapshot_ts=0 batch_sched=normal seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered pretrained_agent=${1} task=quadruped_run snapshot_ts=0 batch_sched=normal seed=20 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered_partial pretrained_agent=${1} task=quadruped_run snapshot_ts=2000000 batch_sched=normal seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered_partial pretrained_agent=${1} task=quadruped_run snapshot_ts=2000000 batch_sched=normal seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6" ./urlb_job_git.sh

# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered_partial pretrained_agent=${1} task=quadruped_jump snapshot_ts=2000000 batch_sched=normal seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5 load_only_encoder=true" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered_partial pretrained_agent=${1} task=quadruped_jump snapshot_ts=2000000 batch_sched=normal seed=20 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6 load_only_encoder=true" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered pretrained_agent=${1} task=quadruped_jump snapshot_ts=0 batch_sched=normal seed=16 batch_sched=normal obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5 " ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered pretrained_agent=${1} task=quadruped_jump snapshot_ts=0 seed=20 obs_type=pixels action_repeat=2 batch_sched=normal buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6 " ./urlb_job_git.sh

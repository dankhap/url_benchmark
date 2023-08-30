#!/bin/bash
echo "in bash script"
export PATH=$PATH:/opt/conda/envs/urlb2/bin:/opt/conda/condabin
echo "all arguments: $@"
eval "$(conda shell.bash hook)"
conda activate urlb2

if [ $1 = "fine" ]; then
	python finetune.py group_name=tests frame_stack=1 eval_every_frames= use_wandb=false pretrained_agent=icm task=walker_walk snapshot_ts=0 obs_type=pixels seed=8 num_seed_frames=2000 replay_buffer_num_workers=0 buffer_dir=/code/url_benchmark/icm_walker_pixels_buffer_bak/ load_only_encoder=true dreamer_conf.dreamer.pretrain=$1 dreamer_conf.dreamer.offline_skip_reward=$2
else 
	python finetune.py ${@:2}
fi


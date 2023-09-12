# python finetune.py \
# 	group_name=dreamer_wm_test \
# 	experiment=draemerv3 \
# 	frame_stack=1 \
# 	eval_every_frames=8000 \
# 	use_wandb=true \
#   pretrained_agent=dreamer \
# 	task=walker_walk \
# 	snapshot_ts=0 \
# 	obs_type=pixels \
# 	seed=8 \
# 	num_seed_frames=4000 \
# 	replay_buffer_num_workers=0 \
# 	buffer_dir=/home/daniel/code/url_benchmark/buffers/icm_walker_pixels_buffer_bak/ \
# 	load_only_encoder=true \
# 	dreamer_conf.dreamer.pretrain=10 \
# 	dreamer_conf.dreamer.reward_finetune_steps=5 \
# 	dreamer_conf.dreamer.log_pretrain_every=2 \
# 	dreamer_conf.dreamer.offline_skip_reward=true

python pretrain_dreamer.py \
	group_name=dreamer_wm_test \
	experiment=draemerv3 \
	frame_stack=1 \
	use_wandb=true \
	agent=icm \
	domain=walker \
	obs_type=pixels \
	seed=8 \
	num_seed_frames=2500 \
	replay_buffer_num_workers=0 \
	dreamer_conf.dreamer.pretrain=10 \
	dreamer_conf.dreamer.reward_finetune_steps=0 \
	dreamer_conf.dreamer.log_pretrain_every=0 \
	dreamer_conf.dreamer.offline_skip_reward=false

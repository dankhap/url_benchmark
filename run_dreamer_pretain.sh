
python pretrain_dreamer.py \
	group_name=dreamerv3 \
	experiment=dreamer_wm_pretrain \
	frame_stack=1 \
	use_wandb=true \
	agent=icm \
	domain=walker \
	obs_type=pixels \
	seed=8 \
	num_seed_frames=4000 \
	dreamer_conf.dreamer.pretrain=10 \
	dreamer_conf.dreamer.reward_finetune_steps=0 \
	dreamer_conf.dreamer.log_pretrain_every=0 \
	dreamer_conf.dreamer.offline_skip_reward=true

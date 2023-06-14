
sbatch --export=ALL,A="fine group_name=${1}_walker_run_buffered_fix pretrained_agent=${1} task=walker_run snapshot_ts=0 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_run_buffered_partial_fix pretrained_agent=${1} task=walker_run snapshot_ts=2000000 seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_run_buffered_fix pretrained_agent=${1} task=walker_run snapshot_ts=0 seed=20 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_run_buffered_partial_fix pretrained_agent=${1} task=walker_run snapshot_ts=2000000 seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh

sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_fix pretrained_agent=${1} task=walker_flip snapshot_ts=0 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_partial_fix pretrained_agent=${1} task=walker_flip snapshot_ts=2000000 seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer5 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_fix pretrained_agent=${1} task=walker_flip snapshot_ts=0 seed=20 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4" ./urlb_job_git.sh
sbatch --export=ALL,A="fine group_name=${1}_walker_flip_buffered_partial_fix pretrained_agent=${1} task=walker_flip snapshot_ts=2000000 seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer4 action_repeat=2 load_only_encoder=true" ./urlb_job_git.sh

# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered pretrained_agent=${1} task=quadruped_run snapshot_ts=0 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered pretrained_agent=${1} task=quadruped_run snapshot_ts=0 seed=20 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered_partial pretrained_agent=${1} task=quadruped_run snapshot_ts=2000000 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer5" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_run_buffered_partial pretrained_agent=${1} task=quadruped_run snapshot_ts=2000000 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer6" ./urlb_job_git.sh

# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered_partial pretrained_agent=${1} task=quadruped_jump snapshot_ts=2000000 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer4 load_only_encoder=true" ./urlb_job_git.sh
# sbatch --export=ALL,A="fine group_name=${1}_quadruped_jump_buffered pretrained_agent=${1} task=quadruped_jump snapshot_ts=0 seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer4 " ./urlb_job_git.sh

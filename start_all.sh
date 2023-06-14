sbatch --export=ALL,A="pre group_name=pretrain_${1}_walker_pix_fix agent=${1} domain=walker seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/${1}_walker_pixel_buffer4" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_quad_pix_fix agent=${1} domain=quadruped seed=16 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/${1}_quad_pixel_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_walker_pix_fix agent=${1} domain=walker seed=26 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/${1}_walker_pixel_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_quad_pix_fix agent=${1} domain=quadruped seed=26 obs_type=pixels action_repeat=2 buffer_dir=/code/url_benchmark/${1}_quad_pixel_buffer6" ./urlb_job_git.sh

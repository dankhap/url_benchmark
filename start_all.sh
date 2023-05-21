# sbatch --export=ALL,A="pre group_name=pretrain_${1}_walker_pix2 agent=${1} domain=walker seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/${1}_walker_pixel_buffer3" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_quad_pix4 agent=${1} domain=quadruped seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/${1}_quad_pixel_buffer4" ./urlb_job_git.sh

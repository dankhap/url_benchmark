sbatch --export=ALL,A="pre group_name=pretrain_icm_walker_pix2 pretrained_agent=icm domain=walker seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_walker_pixel_buffer2" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_icm_quad_pix2 pretrained_agent=icm domain=quadrupted seed=16 obs_type=pixels buffer_dir=/code/url_benchmark/icm_quad_pixel_buffer4" ./urlb_job_git.sh

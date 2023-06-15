sbatch --export=ALL,A="pre group_name=pretrain_${1}_walker_${2} agent=${1} domain=walker seed=16 obs_type_params=${2} buffer_dir=/code/url_benchmark/${1}_walker_${2}_buffer4" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_quadruped_${2} agent=${1} domain=quadruped seed=16 obs_type_params=${2} buffer_dir=/code/url_benchmark/${1}_quad_${2}_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_walker_${2} agent=${1} domain=walker seed=26 obs_type_params=${2} buffer_dir=/code/url_benchmark/${1}_walker_${2}_buffer5" ./urlb_job_git.sh
sbatch --export=ALL,A="pre group_name=pretrain_${1}_quadruped_${2} agent=${1} domain=quadruped seed=26 obs_type_params=${2} buffer_dir=/code/url_benchmark/${1}_quad_${2}_buffer6" ./urlb_job_git.sh

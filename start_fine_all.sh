#!/bin/bash

obs=${1}

for algo in "${@:2}"; do
	bash ./start_fine.sh ${algo} ${obs} walker_run 16 "*"
	bash ./start_fine.sh ${algo} ${obs} walker_run 26 "*"
	bash ./start_fine.sh ${algo} ${obs} walker_flip 16 "*"
	bash ./start_fine.sh ${algo} ${obs} walker_flip 26 "*"

	bash ./start_fine.sh ${algo} ${obs} quadruped_run 16 "*"
	bash ./start_fine.sh ${algo} ${obs} quadruped_run 26 "*"
	bash ./start_fine.sh ${algo} ${obs} quadruped_jump 16 "*"
	bash ./start_fine.sh ${algo} ${obs} quadruped_jump 26 "*"
done


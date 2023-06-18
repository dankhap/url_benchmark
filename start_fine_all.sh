#!/bin/bash

bash ./start_fine.sh re3 states walker run 16
bash ./start_fine.sh re3 states walker run 20
bash ./start_fine.sh re3 states walker flip 16
bash ./start_fine.sh re3 states walker flip 20

bash ./start_fine.sh re3 states quadruped run 16
bash ./start_fine.sh re3 states quadruped run 20
bash ./start_fine.sh re3 states quadruped jump 16
bash ./start_fine.sh re3 states quadruped jump 20

bash ./start_fine.sh re3 pixels walker run 16
bash ./start_fine.sh re3 pixels walker run 20
bash ./start_fine.sh re3 pixels walker flip 16
bash ./start_fine.sh re3 pixels walker flip 20

bash ./start_fine.sh re3 pixels quadruped run 16
bash ./start_fine.sh re3 pixels quadruped run 20
bash ./start_fine.sh re3 pixels quadruped jump 16
bash ./start_fine.sh re3 pixels quadruped jump 20

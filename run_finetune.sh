#!/bin/bash


while [[ $1 != *@* ]]; do
  algos+=("$1")
  #echo $1
  shift
done
shift   # Ignore the sentinal

while [[ $1 != *@* ]]; do
  types+=("$1")
  echo $1
  shift
done
shift

while [[ $1 != *@* ]]; do
  obs+=("$1")
  echo $1
  shift
done
shift
tasks=( "$@" )    # What's left


echo "array1: ${algos[@]}"
echo "array2: ${types[@]}"
echo "array3: ${obs[@]}"
echo "array4: ${tasks[@]}"

for a in "${algos[@]}"
do
    for t in "${types[@]}"
    do
        for o in "${obs[@]}"
	do
	    for k in "${tasks[@]}"
 	    do
		#bash ./start_fine.sh $a $o $k $t 16
		echo "running: bash ./start_fine.sh $a $o $k $t "
		bash ./start_fine.sh $a $o $k 16 $t
		bash ./start_fine.sh $a $o $k 26 $t
	    done
	done
    done
done

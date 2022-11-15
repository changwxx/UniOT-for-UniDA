#!/bin/bash
#SBATCH -J officehome
#SBATCH -p ShangHAI
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=all
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time 2-00:00


cd ..
py_main='main'
gpu=$CUDA_VISIBLE_DEVICES

dataset='officehome'
domains=(Art Clipart Product RealWorld)
exp=${dataset}

for source in ${domains[@]}
do
    for target in ${domains[@]}
    do
        if [[ "${source}" != "${target}" ]]
        then
            python ${py_main}.py --gpu ${gpu} --exp ${exp} --dataset ${dataset} --source ${source} --target ${target}
        fi
    done
done

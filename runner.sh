#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name="fig3"                   # job name
#SBATCH --output=/scratch/%u/single-gpu-fedavg/%x.out   # Output file
#SBATCH --error=/scratch/%u/single-gpu-fedavg/%x.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # number of cores needed
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --time=2-22:00:00                   # set to 2 days 10 h; choose carefully

../federated_participation/rec/bin/python main.py --eta0 0.01 --algorithm scaffold --init_corrections --world-size 8 --communication-interval 32 --rounds 6000 --batchsize 625 --num-evals 100 --scaffold-comparison --dataset CIFAR10 --full_batch --individual_evals --model CNN --layers 2 --loss cross_entropy --seed 0 --heterogeneity 0.9 --log-folder logs --log-fpath sgputest


python3.9 main.py --eta0 0.01 --algorithm scaffold --init_corrections --world-size 8 --communication-interval 32 --rounds 500 --batchsize 625 --num-evals 50 --scaffold-comparison --dataset MNIST --full_batch --individual_evals --model CNN --layers 5 --loss cross_entropy --seed 0 --heterogeneity 0.9 --log-folder logs --log-fpath layer5scaffoldnew.json

python3.9 main.py --eta0 0.01 --algorithm scaffold --init_corrections --world-size 8 --communication-interval 32 --rounds 500 --batchsize 625 --num-evals 50 --scaffold-comparison --dataset MNIST --full_batch --individual_evals --model CNN --layers 7 --loss cross_entropy --seed 0 --heterogeneity 0.9 --log-folder logs --log-fpath layer7scaffoldnew.json



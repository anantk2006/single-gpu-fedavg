#!/usr/bin/env bash

# Common arguments (kept constant)
BASE_ARGS="python3.9 main.py \
  --eta0 0.01 \
  --world-size 8 \
  --rounds 3000 \
  --batchsize 625 \
  --num-evals 100 \
  --dataset CIFAR10 \
  --full_batch \
  --individual_evals \
  --model CNN \
  --layers 3 \
  --loss cross_entropy \
  --seed 0 \
  --log-folder logs"

ALGORITHMS=("scaffold")

############################################
# Sweep 1: Communication Interval
# Hold heterogeneity = 0.5
############################################
COMM_INTERVALS=(8 32 128)
FIXED_HETERO=0.5

#for ALG in "${ALGORITHMS[@]}"; do
#  for COMM in "${COMM_INTERVALS[@]}"; do
#    LOG_FPATH="comm_${COMM}_hetero_${FIXED_HETERO}_${ALG}.json"
#
#    $BASE_ARGS \
#      --algorithm "$ALG" \
#      --communication-interval "$COMM" \
#      --heterogeneity "$FIXED_HETERO" \
#      --log-fpath "$LOG_FPATH"
#  done
#done

############################################
# Sweep 2: Heterogeneity
# Hold communication interval = 32
############################################
HETEROS=(0.1 0.5 0.9)
FIXED_COMM=32

for ALG in "${ALGORITHMS[@]}"; do
  for HET in "${HETEROS[@]}"; do
    LOG_FPATH="comm_${FIXED_COMM}_hetero_${HET}_${ALG}.json"

    $BASE_ARGS \
      --algorithm "$ALG" \
      --communication-interval "$FIXED_COMM" \
      --heterogeneity "$HET" \
      --log-fpath "$LOG_FPATH"
  done
done




#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=8 # number of nodes to use, 2 p4d(e) = 16 A100 GPUs
#SBATCH --job-name=smpv2_llama # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1

set -ex;

###########################
###### User Variables #####
###########################

#########################
model_type=llama_v2
model_size=$1
weights=/fsx/jason-lee/ckpt/sftv4-4nodes-bsz1-exp004-safetensors-v2
out_dir=/fsx/jason-lee/log/03_2024/22-test2
scr_dir=/fsx/jason-lee/scr/smpv2_llama
pretok_path=/fsx/jason-lee/data/03_2024/26_sftv4.001-raw.pt

mkdir -p $out_dir

#Toggle this to use synthetic data
use_synthetic_data=0


# To run training on your own data  set Training/Test Data path  -> Change this to the tokenized dataset path in Fsx. Acceptable formats are huggingface (arrow) and Jsonlines.

export TRAINING_DIR=/fsx/PATH_TO_TRAINING
export TEST_DIR=/fsx/PATH_TO_TEST
export CHECKPOINT_PATH=/fsx/PATH_TO_CHECKPOINT

# default variables for Enroot
: "${IMAGE:=$(pwd)/smpv2.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"
: "${HYPERPOD_PATH:="/var/log/aws/clusters":"/var/log/aws/clusters"}"
: "${TRAIN_DATA_PATH:=$TRAINING_DIR:$TRAINING_DIR}"
: "${TEST_DATA_PATH:=$TEST_DIR:$TEST_DIR}"
: "${CHECKPOINT_PATH:=$CHECKPOINT_PATH:$CHECKPOINT_PATH}"
: "${PRETOK_DATA_PATH:=$pretok_path}"
############


###############
## Environment Variables ##
###########################

#export NCCL_SOCKET_IFNAME=en
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"
export SM_NUM_GPUS=8
export GPU_NUM_DEVICES=8
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0


# async runtime error ...
export CUDA_DEVICE_MAX_CONNECTIONS=1

#########################
## Command and Options ##



if [ "$model_size" == "7b" ]; then
    HIDDEN_WIDTH=4096
    NUM_LAYERS=32
    NUM_HEADS=32
    LLAMA_INTERMEDIATE_SIZE=11008
    DEFAULT_SHARD_DEGREE=8
elif [ "$model_size" == "13b" ]; then
    HIDDEN_WIDTH=5120
    NUM_LAYERS=40
    NUM_HEADS=40
    LLAMA_INTERMEDIATE_SIZE=13760
    # Reduce for better perf on p4de
    DEFAULT_SHARD_DEGREE=64
elif [ "$model_size" == "20b" ]; then
    if [ "$model_type" == "llama_v2" ]; then
        echo "Llama V2 is only configured for 7b, 13b and 70b, please add the configuration if you wish to run 20b"
        exit 1
    fi
    HIDDEN_WIDTH=6144
    NUM_LAYERS=44
    NUM_HEADS=64
    # Reduce for better perf on p4de
    DEFAULT_SHARD_DEGREE=64
elif [ "$model_size" == "65b" ]; then
    if [ "$model_type" == "llama_v2" ]; then
        echo "Llama V2 is only configured for 7b, 13b and 70b, please add the configuration if you wish to run 65b"
        exit 1
    fi
    HIDDEN_WIDTH=8192
    NUM_LAYERS=80
    NUM_HEADS=64
    # Reduce for better perf on p4de
    DEFAULT_SHARD_DEGREE=128
elif [ "$model_size" == "70b" ]; then
    HIDDEN_WIDTH=8192
    NUM_LAYERS=80
    NUM_HEADS=64
    LLAMA_INTERMEDIATE_SIZE=28672
    # Reduce for better perf on p4de
    DEFAULT_SHARD_DEGREE=64
fi

DEFAULT_SHARD_DEGREE=$2


if [ -z "$shard_degree" ]; then
    SHARD_DEGREE=$DEFAULT_SHARD_DEGREE
else
    SHARD_DEGREE=$shard_degree
fi

if [ -z "$LLAMA_INTERMEDIATE_SIZE" ]; then
    LLAMA_ARGS=""
else
    LLAMA_ARGS="--llama_intermediate_size $LLAMA_INTERMEDIATE_SIZE "
fi


declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $HYPERPOD_PATH,$FSX_MOUNT,$weights,$out_dir,$scr_dir,$PRETOK_DATA_PATH
)

declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname) \
)

srun -l "${ARGS[@]}" torchrun "${TORCHRUN_ARGS[@]}" $scr_dir/scripts/train_external.py \
            --train_batch_size 1 \
            --max_steps 10000 \
            --hidden_width $HIDDEN_WIDTH \
            --num_layers $NUM_LAYERS \
            --num_heads $NUM_HEADS \
            ${LLAMA_ARGS} \
            --shard_degree $SHARD_DEGREE \
            --model_type $model_type \
            --profile_nsys 1 \
            --use_smp_implementation 1 \
            --max_context_width 16384 \
            --tensor_parallel_degree 1 \
            --fp8 $3 \
            --use_synthetic_data $use_synthetic_data \
            --pretokenized_path $PRETOK_DATA_PATH \

            # --hf_pretrained_model_name_or_dir $weights \

            #--checkpoint_dir $out_dir \
            # --training_dir $TRAIN_DATA_PATH \
            # --test_dir $TEST_DATA_PATH \
            # --dataset_type hf \
            # --checkpoint_dir $CHECKPOINT_PATH
            # --checkpoint_freq 500 \
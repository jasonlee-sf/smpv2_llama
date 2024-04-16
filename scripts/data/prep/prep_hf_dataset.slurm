#!/bin/bash
#SBATCH --output=logs/%x_%j.out  # Redirects outputs to file in current_dir/logs
#SBATCH --error=logs/%x_%j.out  # Redirects err to same file in current_dir/logs
#SBATCH --job-name=prep_hf_data
#SBATCH --ntasks-per-node=1
#SBATCH -N 1

: "${IMAGE:=/fsx/jason-lee/scr/smpv2_llama/smpv2.sqsh}"
: "${HYPERPOD_PATH:="/var/log/aws/clusters":"/var/log/aws/clusters"}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"

OUTPUT_DIR=/fsx/jason-lee/data/wikicorpus__raw_en/llama/4096/
mkdir -p $OUTPUT_DIR
TOKENIZER=Salesforce/xgen2-oai

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $HYPERPOD_PATH,$FSX_MOUNT,$OUTPUT_DIR
)

## Below examples for llama tokenizer

srun -l "${ARGS[@]}" python data/prep/prepare_hf_dataset.py \
    --dataset_name wikicorpus \
    --dataset_config_name raw_en \
    --val_split_percentage 10 \
    --hf_tokenizer_name $TOKENIZER \
    --seq_len 16384 \
    --output_dir $OUTPUT_DIR

## C4
# Had to delete a file which was incomplete and crashed the job
# rm /fsx/datasets/.cache/datasets/downloads/extracted/741a4aaf04e7748f791ce4525c5876f13a45e8115d76b099c818cf7970972c48
# srun -l "${ARGS[@]}" python data/prep/prepare_hf_dataset.py --dataset_path /fsx/datasets/c4/en/hf \
#     --output_dir /fsx/datasets/temp/c4/en/hf-tokenized/llama \
#     --hf_tokenizer_name meta-llama/Llama-2-7b-hf \
#     --seq_len 4096 \
#     --val_split_percentage 20

#!/bin/bash
#SBATCH --job-name=seq_evo2
#SBATCH --output=logs/seq_evo2_%A_%a.out
#SBATCH --error=logs/seq_evo2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# Create logs directory if it doesn't exist
mkdir -p logs

# Define Evo2 models and data types
MODEL_NAMES=(
    "evo2_1b_base"
    "evo2_7b_base"
    "evo2_1b_base"
    "evo2_7b_base"
)

DATA_TYPES=(
    "eukaryote"
    "eukaryote"
    "bacteria"
    "bacteria"
)

# Get current model and data type from array task ID
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
DATA_TYPE=${DATA_TYPES[$SLURM_ARRAY_TASK_ID]}

# Set output directory based on data type
OUTPUT_DIR="./sequence_recovery_results/${DATA_TYPE}"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Running Evo2 Sequence Recovery Benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_NAME}"
echo "Data Type: ${DATA_TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run the Evo2 sequence recovery script with optimized batch size
# Uses batch_size=32 (default) with Transformer Engine + Flash Attention
python src/tasks/downstream/sequence_recovery_evo2.py \
    --data_type ${DATA_TYPE} \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Job completed!"
echo "=========================================="

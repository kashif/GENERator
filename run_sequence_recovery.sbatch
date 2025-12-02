#!/bin/bash
#SBATCH --job-name=seq_recovery
#SBATCH --output=logs/seq_recovery_%A_%a.out
#SBATCH --error=logs/seq_recovery_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-7

# Create logs directory if it doesn't exist
mkdir -p logs

# Define models and data types
# Run both v1 and v2 GENERator models on both eukaryote and bacteria
MODELS=(
    "GenerTeam/GENERator-eukaryote-1.2b-base"
    "GenerTeam/GENERator-eukaryote-3b-base"
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
    "GenerTeam/GENERator-v2-eukaryote-3b-base"
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base"
    "GenerTeam/GENERator-v2-prokaryote-3b-base"
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base"
    "GenerTeam/GENERator-v2-prokaryote-3b-base"
)

DATA_TYPES=(
    "eukaryote"
    "eukaryote"
    "eukaryote"
    "eukaryote"
    "bacteria"
    "bacteria"
    "bacteria"
    "bacteria"
)

MODEL_NAMES=(
    "GENERator-1B"
    "GENERator-3B"
    "GENERator-v2-1B"
    "GENERator-v2-3B"
    "GENERator-v2-1B"
    "GENERator-v2-3B"
    "GENERator-v2-1B"
    "GENERator-v2-3B"
)

# Get current model and data type from array task ID
MODEL_PATH=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATA_TYPE=${DATA_TYPES[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

# Set output directory based on data type
OUTPUT_DIR="./sequence_recovery_results/${DATA_TYPE}"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Running Sequence Recovery Benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_PATH}"
echo "Data Type: ${DATA_TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run the sequence recovery script
python src/tasks/downstream/sequence_recovery.py \
    --data_type ${DATA_TYPE} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --bf16

echo "=========================================="
echo "Job completed!"
echo "=========================================="

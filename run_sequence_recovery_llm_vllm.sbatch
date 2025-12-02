#!/bin/bash
#SBATCH --job-name=seq_llm_vllm
#SBATCH --output=logs/seq_llm_vllm_%A_%a.out
#SBATCH --error=logs/seq_llm_vllm_%A_%a.err
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5

# Create logs directory if it doesn't exist
mkdir -p logs

# Define model-organism combinations
# Array index 0-5 for 3 models x 2 organisms
MODELS=(
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-30B-A3B-Base"
    "Qwen/Qwen3-30B-A3B-Base"
)

DATA_TYPES=(
    "bacteria"
    "eukaryote"
    "bacteria"
    "eukaryote"
    "bacteria"
    "eukaryote"
)

# Get model and data type for this task
MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATA_TYPE=${DATA_TYPES[$SLURM_ARRAY_TASK_ID]}

# Set batch size based on model size (vLLM can handle much larger batches)
if [[ "$MODEL_NAME" == *"30B"* ]]; then
    BATCH_SIZE=32
elif [[ "$MODEL_NAME" == *"14B"* ]]; then
    BATCH_SIZE=64
else
    BATCH_SIZE=128
fi

# Output directory
OUTPUT_DIR="./sequence_recovery_results"

echo "=========================================="
echo "SLURM Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_NAME"
echo "Data Type: $DATA_TYPE"
echo "Batch Size: $BATCH_SIZE (vLLM)"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Run the vLLM sequence recovery script
python src/tasks/downstream/sequence_recovery_llm.py \
    --data_type ${DATA_TYPE} \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Job completed for ${MODEL_NAME} on ${DATA_TYPE}"
echo "=========================================="

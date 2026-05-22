#!/bin/bash -l
#SBATCH --job-name=feature_extraction
#SBATCH --output=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.out
#SBATCH --error=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mhaiderzaidi21@fau.de

# Resource allocation (TinyX specific)
#SBATCH --partition=v100          # or 'gpu_test' for quick tests (<30min)
#SBATCH --gres=gpu:v100:1       # Request 1 A100 GPU (adjust type if needed)
#SBATCH --time=04:00:00          # Wall time (adjust based on sample count)



echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"

cd $WORK/BIMAP-FEATURE


# ==========================================
# Environment Setup (TinyX specific modules)
# ==========================================

# Load necessary modules
module purge
#module load python/3.10                # or latest available
module load cuda/12.2                  # Match your PyTorch CUDA version
#module load cudnn/8.9                  # cuDNN for GPU acceleration


source $WORK/.conda/bin/activate feature  

# Show GPU info
echo "CUDA available devices:"
nvidia-smi


export HF_HOME=$WORK/.cache/huggingface
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers

pip install torch

OUTPUT_DIR=/home/woody/vlbi/vlbi108v/BIMAP-FEATURE/embeddings


python scripts/extract_features.py \
    --model facebook/wav2vec2-base \
    --languages en_us de_de es_419 \
    --max-samples 100 \
    --output-dir $OUTPUT_DIR \
    --device cuda \
    --split train



echo "Job finished at: $(date)"
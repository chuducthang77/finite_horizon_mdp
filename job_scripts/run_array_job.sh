#!/bin/bash
#SBATCH --account=def-szepesva       # Replace with your allocation
#SBATCH --job-name=my_array_job
#SBATCH --time=00:05:00                 # hh:mm:ss
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-49                     # 10 jobs, indices 0 through 9
#SBATCH --output=logs/%x_%A_%a.out      # %A=job ID, %a=array index
#SBATCH --error=logs/%x_%A_%a.err

# Optional: load modules
module load python/3.11

# Create output dir if not exist
mkdir -p ../logs

# Print info
learning_rates=($(python -c "import numpy as np; print(' '.join(map(str, np.exp(np.linspace(-9, 0, 50)))))"))
LR=${learning_rates[$SLURM_ARRAY_TASK_ID]}

echo "Running task ${SLURM_ARRAY_TASK_ID} with learning rate ${LR}"

# Run your job script
cd ..
python chain_mdp_training.py --lr ${LR}

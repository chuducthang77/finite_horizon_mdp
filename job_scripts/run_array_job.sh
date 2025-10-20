#!/bin/bash
#SBATCH --account=def-szepesva       # Replace with your allocation
#SBATCH --time=00:05:00                 # hh:mm:ss
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99                     # 10 jobs, indices 0 through 9
#SBATCH --output=logs/%x_%A_%a.out      # %A=job ID, %a=array index
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=thang@ualberta.ca
#SBATCH --mail-type=ALL

# Optional: load modules
module load python/3.11 scipy-stack
source ~/venv/bin/activate

# Create output dir if not exist
mkdir -p ../logs

# Print info
learning_rates=($(python -c "import numpy as np; print(' '.join(map(str, np.exp(np.linspace(-9, 0, 100)))))"))
LR=${learning_rates[$SLURM_ARRAY_TASK_ID]}

echo "Running task ${SLURM_ARRAY_TASK_ID} with learning rate ${LR}"

# Run your job script
cd ..
python chain_mdp_training.py --lr ${LR} --title "chain_length_4"
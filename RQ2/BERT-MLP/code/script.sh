## CC Script
module load StdEnv/2023
module load python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install numpy pandas scikit-learn torch transformers wandb --no-index
wandb login <YOUR_API_KEY>
python train.py
deactivate
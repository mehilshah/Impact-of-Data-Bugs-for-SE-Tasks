## CC Script
module load StdEnv/2023
module load python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install nltk numpy==1.23.2+computecanada scikit_learn torch tqdm transformers wandb --no-index
wandb login <YOUR_API_KEY>
python main.py -train -train_data=./data/qt_data/qt_train_changed.pkl -save-dir=./ -dictionary_data=./data/qt_dict.pkl
deactivate
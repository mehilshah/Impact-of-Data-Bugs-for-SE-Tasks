
# Vulnerability Detection using CodeBERT

This repository contains the code for training and evaluating the CodeBERT model for vulnerability detection in source code. The project focuses on using the Devign dataset to train a deep learning model for identifying vulnerabilities in code.

## Repository Structure

└── vulnerability-detection
    ├── LICENSE
    ├── codebert
    │   ├── model.py
    │   ├── run.py
    │   └── script.sh
    └── requirements.txt

## Components

1.  **codebert/**: Contains the core implementation of the CodeBERT model and training scripts.
    
    -   **model.py**: Defines the CodeBERT model architecture.
        
    -   **run.py**: Main script to train and evaluate the model.
        
    -   **script.sh**: Shell script to automate the training process.
        
2.  **requirements.txt**: Lists the Python dependencies required to run the code.
    

## Prerequisites

Before running the experiments, ensure you have:

1.  Python 3.7+ installed.
    
2.  Required dependencies installed. You can install them using:
    
    pip install -r requirements.txt
    
3.  The Devign dataset prepared and placed in the appropriate directory (refer to the dataset preparation section below).
    

## Dataset Preparation

Use the same methodology from the other baseline (LineVul) to prepare the data and place it in the appropriate folder.
    

## Running Experiments

To train the CodeBERT model on the Devign dataset, use the following command:

python run.py --do_train --training standard --data_root devign --project_name qemu --epoch 20 --seed 123456

### Command Arguments

-   `--do_train`: Flag to indicate training mode.
    
-   `--training`: Specifies the training method (e.g.,  `standard`).
    
-   `--data_root`: Directory containing the dataset (e.g.,  `devign`).
    
-   `--project_name`: Name of the project (e.g.,  `qemu`).
    
-   `--epoch`: Number of training epochs.
    
-   `--seed`: Random seed for reproducibility.


## Acknowledgments

This research is based on the work presented by the authors of the paper at the  [Vulnerability detection with code language models: How far are we?](https://arxiv.org/abs/2403.18624), which will be presented at ICSE 2025. We thank the authors of the paper for implementation of the baseline techniques.

For more details on the baseline techniques and initial setup, please refer to the original paper and associated documentation.
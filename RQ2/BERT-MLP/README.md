
# Impact of Data Bugs on Deep Learning Models in Software Engineering

This repository contains the code for analyzing the impact of data quality issues, specifically  **label noise**,  **class imbalance**, and  **concept drift**, on deep learning models for duplicate bug report detection. The study uses BERT-based models and MLP classifiers to evaluate the effects of these data bugs on model performance.

## Repository Structure
.
├── data/                     # Contains the dataset files
│   ├── ThunderBird/          # ThunderBird dataset (duplicate and non-duplicate bug reports)
│   ├── ThunderBird-old/      # Older version of ThunderBird dataset for concept drift analysis
├── src/                      # Source code for the experiments
│   ├── main.py               # Main script to run the experiments
├── README.md                 # This file
├── requirements.txt          # Python dependencies

## Running Experiments

To run the main data quality analysis, use the following command:

python src/main.py --label-noise --class-imbalance

### Command-Line Arguments

-   `--label-noise`: Clean label noise in the dataset
    
-   `--class-imbalance`: Handle class imbalance in the 
    

### Concept Drift Analysis

For  **concept drift**  analysis, the older version of the dataset (`ThunderBird-old`) is used. This dataset is automatically loaded and processed when running the experiments. No additional command-line argument is required for concept drift analysis.

### Example Commands
 python src/main.py (with the appropriate commands, -label-noise, -class-imbalance, or running the script with the old dataset loaded for concept drift analysis)

## Model Training and Evaluation

The pipeline includes:

1.  **BERT-based feature extraction**: Extracts features from bug report text using a pre-trained BERT model.
    
2.  **MLP Classifier**: Trains an MLP classifier using  `RandomizedSearchCV`  for hyperparameter tuning.
    
3.  **Evaluation**: Evaluates the model using classification metrics (e.g., precision, recall, F1-score) and logs results to Weights & Biases (W&B).
    

## Results

The results of the experiments, including classification reports and confusion matrices, are logged to W&B for further analysis. You can visualize the results in the W&B dashboard.

## Contact

For questions or further information, please contact: shahmehil@dal.ca

## Acknowledgments

This research builds on the work presented in the paper  ### [Duplicate bug report detection  using an  attention-based neural language model](https://ieeexplore.ieee.org/abstract/document/9852720/?casa_token=2UFugZFmrlkAAAAA:kKYJOVNEc7xpJDONu3rLVtQOTXckG2G-3pENQxgqmeVy00Fv3wqB6pBZJL9wxsaNNHEvSVzI). We thank the authors for their baseline techniques and the initial dataset used in our experiments.

For more details on the baseline techniques and initial setup, please refer to the original paper and its associated resources.
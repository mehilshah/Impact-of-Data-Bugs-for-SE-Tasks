
# Impact of Data Bugs on Deep Learning Models in Software Engineering
This folder contains the code for the RQ1 of our study, which focuses on the impact of data quality issues in code-based data, using vulnerability detection as the downstream task.

## Repository Structure

```
.
├── Attention_Weights_Analysis/
│   ├── Bugfree-Attention-Weights/
│   │   ├── BigVul/
│   │   └── Devign/
│   └── Buggy-Attention-Weights/
│       ├── BigVul/
│       └── Devign/
├── dq_analysis/
│   ├── attributes/
│   │   ├── accuracy.py
│   │   ├── completeness.py
│   │   ├── consistency.py
│   │   ├── currency.py
│   │   ├── data.py
│   │   ├── __init__.py
│   │   └── uniqueness.py
│   ├── datasets/
│   │   ├── data.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── svp/
│       ├── dq_controller.py
│       ├── __init__.py
│       ├── LineVul/
│       └── run.py
├── measure.sh
├── README_ICSE23.md
├── README.md
└── setup.py

```

## Components

1.  **Attention_Weights_Analysis/**: Contains attention weights for both bug-free and buggy models, categorized by datasets (BigVul and Devign).
2.  **dq_analysis/**: Core package for data quality analysis.
    -   **attributes/**: Modules for assessing and analyzing the various data quality attributes.
    -   **datasets/**: Dataset handling and processing.
    -   **svp/**: Software Vulnerability Prediction (SVP) specific modules.
        -   **dq_controller.py**: Main controller for data quality experiments.
        -   **LineVul/**: LineVul model implementation.
        -   **run**: Python script to train the LineVul model.
3.  **measure**: Shell script for measuring the data quality issues in the existing datasets.
4.  **setup**: Setup script for the project.

## Prerequisites

Before running the experiments, ensure you have:

1.  Python 3.7+ installed
2.  Required dependencies (listed in  `setup.py`)
3.  Datasets: BigVul and Devign (refer to  `README_ICSE23.md`  for dataset preparation)


## Running Experiments

To run the main data quality analysis:
        ```
    python dq_analysis/svp/dq_controller.py <dataset> 
    ```
    
  Replace  `<dataset>`  with either  `BigVul`  or  `Devign`.
    
## Contact
For questions or further information, please contact: shahmehil@dal.ca

## Acknowledgments

This research is based on the work presented by the authors of the paper at the [Data quality for software vulnerability datasets](https://ieeexplore.ieee.org/abstract/document/10172650/), which was presented at ICSE 2023. We thank the authors of the paper for the baseline techniques and their analysis of data quality issues.

For more details on the baseline techniques and initial setup, please refer to  `README_ICSE23.md`.
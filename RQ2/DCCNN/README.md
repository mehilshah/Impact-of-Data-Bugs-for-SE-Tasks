
# Impact of Data Bugs on Deep Learning Models in Software Engineering
This folder contains the code for the RQ2 of our study, which focuses on the impact of data quality issues in text-based data, using duplicate bug report detection as the downstream task.

## Repository Structure
.
├── DC-CNN
├── HINDBR
├── README.md
├── README_TOSEM.md
├── SABD
└── t-SNE Plots - contains the results for the XAI analysis.

Refer to the individual README of the techniques, as provided by the authors of the TOSEM-DBRD for the processing steps.

## Running Experiments

To run the main data quality analysis:
    ```
    python DCCNN.py --project $PROJECT
    ```

For concept drift, use the project <project>-old from the original dataset.
For class imbalance and concept drift, you can use the parameters --label-noise, and --class-imbalance with the command.
    
## Contact
For questions or further information, please contact: shahmehil@dal.ca

## Acknowledgments

This research is based on the work presented by the authors of the papers [Duplicate Bug Report Detection: How Far Are We?](https://dl.acm.org/doi/full/10.1145/3576042). We thank the authors of the paper for the baseline techniques, and the initial data required for our experiment.

For more details on the baseline techniques and initial setup, please refer to `README_TOSEM.md`.
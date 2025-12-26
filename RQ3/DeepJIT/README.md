# Impact of Data Bugs on Deep Learning Models in Software Engineering
This folder contains the code for the RQ3 of our study, which focuses on the impact of data quality issues in metric-based data, using defect prediction as the downstream task.

## Running Experiments

To run the main data quality analysis:
    ```
      python main.py -train -train_data ./data/<dataset>/<dataset>_train.pkl -dictionary_data ./data/<dataset>/<dataset>_dict.pkl -<dq>
    ```

For concept drift, you can generate the new data using the data extraction module from the work of Zeng et al., ISSTA'21 (https://dl.acm.org/doi/10.1145/3460319.3464819), once the data has been generated, you can place them in the data folder and run the scripts.

For class imbalance and concept drift, you can use the parameters --label-noise, and --class-imbalance with the command.
    
## Contact
For questions or further information, please contact: shahmehil@dal.ca

## Acknowledgments

This research is based on the work presented by the authors of the papers (Deep just-in-time defect prediction: how far are we?
)[https://dl.acm.org/doi/10.1145/3460319.3464819], and (Deepjit: an end-to-end deep learning framework for just-in-time defect prediction)[https://ieeexplore.ieee.org/abstract/document/8816772?casa_token=cOQO2TcuEPAAAAAA:MUQLr_zfFBuOvWau0MQEVWAzmDZ-dmRWFONWVYUYQSQLhspn2AZTlMtDZkFFh4Ym-xbZYuPhDmU]. We thank the authors of the paper for the baseline techniques, and the initial data required for our experiment.

For more details on the baseline techniques, datasets used and initial setup, please refer to `README_ISSTA21.md`.

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

This research is based on the work presented by the authors of the papers [A study on the  impact of  pre-trained model on  Just-In-Time defect prediction](https://ieeexplore.ieee.org/abstract/document/10366735/?casa_token=nJgvfetVYZIAAAAA:BFfKu99nw-zeRyL9BWZby-l8NSdpcFCqNYcemuVtqqqtWHL5YmnKbNbpIVjZlObpdmsGVHPO). We thank the authors of the paper for the baseline techniques, and the initial data required for our experiment.
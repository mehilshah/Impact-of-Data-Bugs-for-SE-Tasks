"""
Helper functions to run SOTA models for impact analysis.
Alter the train, val, test datasets.
"""
import os
import subprocess
import pandas as pd
import random


# Check if a results file exists
if not os.path.isfile('./results/results.csv'):
    results = pd.DataFrame(columns=['Dataset', 'Model', 'Experiment', 'Precision', 'Recall', 'F1', 'MCC'])
    results.to_csv('./results/results.csv', index=False)


# Generate a random string for temporary file allocation
def get_random_string(length=9):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    random.seed()
    return ''.join(random.choice(chars) for i in range(length))


def run_linevul(train, val, test, dataset, load, save):
    """
    Store datasets to a temporary location.
    Run LineVul using the specified parameters in their reproduction package.
    """

    # Make a temporary location for dataset storage
    temp_loc = get_random_string()
    datadir = f'{os.getcwd()}/dq_analysis/svp/{temp_loc}/'
    print(datadir)
    p = subprocess.Popen(f"mkdir {datadir}", shell=True)
    p.wait()

    # Store datasets
    train.to_csv(datadir+'train.csv', index=False)
    val.to_csv(datadir+'val.csv', index=False)
    test.to_csv(datadir+'test.csv', index=False)
    
    # Verification
    train_verifier = pd.read_csv(f'{os.getcwd()}/dq_analysis/svp/{temp_loc}/train.csv')
    print (f'Train Set: {train_verifier.size}')

    val_verifier = pd.read_csv(f'{os.getcwd()}/dq_analysis/svp/{temp_loc}/val.csv')
    print (f'Validation Set: {val.size}')

    test_verifier = pd.read_csv(f'{os.getcwd()}/dq_analysis/svp/{temp_loc}/test.csv')
    print (f'Test Set: {test_verifier.size}')

    if not load:
        command = 'python linevul_main.py ' + \
            f'--output_dir={os.getcwd()}/dq_analysis/svp/{temp_loc} ' + \
            '--model_type=roberta ' + \
            '--tokenizer_name=microsoft/codebert-base ' + \
            '--model_name_or_path=microsoft/codebert-base ' + \
            '--do_train ' + \
            '--do_test ' + \
            '--reasoning_method attention ' + \
            '--write_raw_preds ' + \
            f'--train_data_file={os.getcwd()}/dq_analysis/svp/{temp_loc}/train.csv ' + \
            f'--eval_data_file={os.getcwd()}/dq_analysis/svp/{temp_loc}/val.csv ' + \
            f'--test_data_file={os.getcwd()}/dq_analysis/svp/{temp_loc}/test.csv ' + \
            '--epochs 5 ' + \
            '--block_size 512 ' + \
            '--train_batch_size 8 ' + \
            '--eval_batch_size 8 ' + \
            '--learning_rate 2e-5 ' + \
            '--max_grad_norm 1.0 ' + \
            '--evaluate_during_training ' + \
            '--seed 123456'
    else:
        command = 'python linevul_main.py ' + \
            f'--output_dir=../../../../dq_analysis/svp/saved_models/{dataset}/ ' + \
            '--model_type=roberta ' + \
            '--tokenizer_name=microsoft/codebert-base ' + \
            '--model_name_or_path=microsoft/codebert-base ' + \
            '--do_test ' + \
            '--write_raw_preds ' + \
            f'--train_data_file=../../../../{datadir}train.csv ' + \
            f'--eval_data_file=../../../../{datadir}val.csv ' + \
            f'--test_data_file=../../../../{datadir}test.csv ' + \
            '--block_size 512 ' + \
            '--eval_batch_size 128'

    with subprocess.Popen(command, cwd='dq_analysis/svp/LineVul/linevul',
                          shell=True, stdout=subprocess.PIPE) as proc:
        output = [x.decode("utf-8") for x in proc.stdout.readlines()]
        metrics = {}
        for result in output:
            print("Result: " + result)
            if 'test_recall' in result:
                metrics['Recall'] = float(result.split(' = ')[1].rstrip())
            if 'test_precision' in result:
                metrics['Precision'] = float(result.split(' = ')[1].rstrip())
            if 'test_f1' in result:
                metrics['F1'] = float(result.split(' = ')[1].rstrip())
            if 'test_mcc' in result:
                metrics['MCC'] = float(result.split(' = ')[1].rstrip())

    # Cleanup
    if save:
        if not os.path.isdir(f"dq_analysis/svp/saved_models/{dataset}/"):
            # Check if dir exists
            os.mkdir(f"dq_analysis/svp/saved_models/{dataset}/")
            os.mkdir(f"dq_analysis/svp/saved_models/{dataset}/checkpoint-best-f1/")
        p = subprocess.Popen(f"mv {datadir}checkpoint-best-f1/model.bin dq_analysis/svp/saved_models/{dataset}/checkpoint-best-f1/", shell=True)
        p.wait()
    p = subprocess.Popen(f"rm -r {datadir}", shell=True)

    return metrics


def run_sota(train, val, test, model, dataset, experiment, load=False, save=False):
    """
    Run and evaluate a specified SOTA model.
    """

    # Define save location
    model_dir = dataset if 'currency' not in experiment else dataset+'_currency'

    run_model = {
        'linevul': run_linevul,
    }
    metrics = run_model[model](train, val, test, model_dir, load, save)
    print(metrics)

    # Store the results
    metrics['Dataset'] = dataset
    metrics['Model'] = model
    metrics['Experiment'] = experiment
    results = pd.DataFrame(columns=['Dataset', 'Model', 'Experiment', 'Precision', 'Recall', 'F1', 'MCC'])
    results.loc[len(results), metrics.keys()] = list(metrics.values())
    results.to_csv('results/results.csv', mode='a', index=False, header=False)

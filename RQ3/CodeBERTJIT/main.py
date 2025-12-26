import argparse
import pickle
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from cleanlab.pruning import get_noise_indices
from evaluation import evaluation_model
from train import train_model
from utils import _read_tsv
from tokenization_of_bert import tokenization_for_codebert

def read_args():
    parser = argparse.ArgumentParser(description='Arguments for training and evaluating the DeepJIT model.')
    parser.add_argument('-train', action='store_true', help='Train the DeepJIT model')
    parser.add_argument('-valid', action='store_true', help='Validate the trained model')
    parser.add_argument('-train_data', type=str, help='Directory of training data')
    parser.add_argument('-dictionary_data', type=str, help='Directory of dictionary data')
    parser.add_argument('-predict', action='store_true', help='Predict using the trained model')
    parser.add_argument('-pred_data', type=str, help='Directory of testing data')
    parser.add_argument('-load_model', type=str, help='Path to the model to be loaded')
    parser.add_argument('-label-noise', action='store_true', help='Clean label noise from data')
    parser.add_argument('-class-imbalance', action='store_true', help='Handle class imbalance using SMOTE')
    parser.add_argument('-msg_length', type=int, default=100, help='Length of the commit message')
    parser.add_argument('-code_length', type=int, default=120, help='Length of each line of commit code')
    parser.add_argument('-batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='Disable GPU usage')
    return parser

def preprocess_data(data, params):
    ids, labels, msgs, codes = data
    if params.label_noise:
        # Assuming a placeholder confident_joint matrix for demonstration
        confident_joint = np.array([[100, 10], [10, 100]])
        noisy_indices = get_noise_indices(s=labels, psx=confident_joint[labels], sorted_index_method='normalized_margin')
        ids, labels, msgs, codes = (np.delete(arr, noisy_indices) for arr in [ids, labels, msgs, codes])

    if params.class_imbalance:
        smote = SMOTE()
        msgs, labels = smote.fit_resample(np.array(msgs).reshape(-1, 1), labels)
        msgs = msgs.flatten()

    return ids, labels, msgs, codes

def load_and_preprocess_data(filepath, params):
    data = pickle.load(open(filepath, 'rb'))
    return preprocess_data(data, params)

def tokenize_data(msgs, codes, params):
    pad_msg = tokenization_for_codebert(msgs, max_length=params.msg_length, flag='msg')
    pad_code = tokenization_for_codebert(codes, max_length=params.code_length, flag='code')
    return pad_msg, pad_code

def main():
    params = read_args().parse_args()
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary

    if params.train or params.valid:
        data = load_and_preprocess_data(params.train_data, params)
        ids, labels, msgs, codes = data
        pad_msg, pad_code = tokenize_data(msgs, codes, params)
        prepared_data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        if params.train:
            train_model(data=prepared_data, params=params)
        else:
            evaluation_model(data=prepared_data, params=params)
    elif params.predict:
        data = load_and_preprocess_data(params.pred_data, params)
        ids, labels, msgs, codes = data
        pad_msg, pad_code = tokenize_data(msgs, codes, params)
        prepared_data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        evaluation_model(data=prepared_data, params=params)
    else:
        print("No operation specified. Use -train, -valid, or -predict.")

if __name__ == '__main__':
    main()
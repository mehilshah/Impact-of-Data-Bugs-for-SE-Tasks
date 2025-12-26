from model import CodeBERT4JIT
from utils import mini_batches, pad_input_matrix
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def preprocess_data(data, params):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    pad_msg = [np.array(component) for component in pad_msg]
    pad_code = [np.array(component) for component in pad_code]
    
    for component in pad_code:
        pad_input_matrix(component, params.code_line)
    
    return pad_msg, pad_code, labels, len(dict_msg), len(dict_code)

def setup_model(params, vocab_msg, vocab_code):
    params.vocab_msg = vocab_msg
    params.vocab_code = vocab_code
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    
    model = CodeBERT4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))
    return model

def evaluate_batches(model, batches):
    model.eval()  # Set model to evaluation mode
    all_predict, all_label = [], []
    
    with torch.no_grad():
        for batch in tqdm(batches):
            batch = [torch.tensor(component).cuda() if torch.cuda.is_available() else torch.tensor(component) for component in batch]
            *inputs, labels = batch
            
            if torch.cuda.is_available():
                labels = labels.cuda().float()
            
            predict = model(*inputs)
            if torch.cuda.is_available():
                predict = predict.cpu()
                
            all_predict.extend(predict.detach().numpy().tolist())
            all_label.extend(labels.tolist())
    
    return all_predict, all_label

def evaluation_model(data, params):
    pad_msg, pad_code, labels, vocab_msg, vocab_code = preprocess_data(data, params)
    model = setup_model(params, vocab_msg, vocab_code)
    batches = mini_batches(*pad_msg, *pad_code, labels, mini_batch_size=params.batch_size)
    all_predict, all_label = evaluate_batches(model, batches)
    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    print('Test data -- AUC score:', auc_score)
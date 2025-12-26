from model import DeepJIT
import torch 
from tqdm import tqdm
from utils import mini_batches_train, save, mini_batches_smote
import torch.nn as nn
import os, datetime
import wandb
from cleanlab.latent_estimation import estimate_latent
from cleanlab.rank import order_label_errors
wandb.init(project = 'case-study-deepjit')

def handle_label_noise(model, data_pad_msg, data_pad_code, data_labels, threshold=0.1):
    # Get model predictions
    predictions = model.forward(torch.tensor(data_pad_msg).cuda(), torch.tensor(data_pad_code).cuda()).cpu().detach().numpy()

    # Estimate latent variables based on labels and predictions
    latent = estimate_latent(s=data_labels, psx=predictions)

    # Order the label errors
    ordered_label_errors = order_label_errors(s=data_labels, psx=predictions, latent=latent)

    error_threshold = int(len(data_labels) * threshold) 
    clean_indices = ordered_label_errors[error_threshold:, 0]  # skip the top 10% most likely errors

    return clean_indices
def train_model(data, params, dq):
    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data
    
    if dq == 'label_noise':
        # customize the threshold to remove more or less noisy labels
        clean_indices = handle_label_noise(model, data_pad_msg, data_pad_code, data_labels, threshold=0.1)
        data_pad_msg = data_pad_msg[clean_indices]
        data_pad_code = data_pad_code[clean_indices]
        data_labels = data_labels[clean_indices]
    
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DeepJIT(args=params)
    wandb.watch(model, log = 'all')
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    criterion = nn.BCELoss()
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        if dq == 'class_imbalance':
            batches = mini_batches_smote(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        else:
            batches = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:            
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))    
        save(model, params.save_dir, 'epoch', epoch)
    
    wandb.finish()
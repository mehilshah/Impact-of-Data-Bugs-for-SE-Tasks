import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import transformers as ppb
import warnings
import logging
from scipy.stats import randint, uniform
import wandb
from imblearn.over_sampling import SMOTE
from cleanlab import classification
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = '../data/ThunderBird/'
MAX_SEQ_LENGTH = 300
BATCH_SIZE = 50
STOP_WORDS = r'i me my myself we our ours ourselves you your yours yourself yourselves they we him he his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now java com org'

def initialize_wandb():
    """Initialize Weights & Biases."""
    wandb.init(project="bert-duplicate-bug-report-detection", config={
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 10
    })

def load_datasets():
    """Load and preprocess datasets."""
    logging.info("Loading datasets...")
    df1 = pd.read_csv(f'{DATA_PATH}dup_TB.csv', delimiter=';')
    df2 = pd.read_csv(f'{DATA_PATH}Nondup_TB.csv', delimiter=';')
    
    df1['Label'] = 'duplicate'
    df2['Label'] = 'non duplicate'
    
    return df1, df2

def remove_stop_words(df1, df2):
    """Remove stop words from the datasets."""
    logging.info("Removing stop words...")
    for df in [df1, df2]:
        for col in ['Title1', 'Title2', 'Description1', 'Description2']:
            df[col] = df[col].str.replace(STOP_WORDS, '')
    return df1, df2

def create_batches(df1, df2):
    """Create batches from the datasets."""
    logging.info("Creating batches...")
    batches = [pd.concat([df1[i:i+BATCH_SIZE], df2[i:i+BATCH_SIZE]], ignore_index=True) 
               for i in range(0, 3486, BATCH_SIZE)]
    test_batches = [pd.concat([df1[i:i+BATCH_SIZE], df2[i:i+BATCH_SIZE]], ignore_index=True) 
                    for i in range(3486, 4374, BATCH_SIZE)]
    return batches, test_batches

def _get_segments(tokens, max_seq_length):
    """Generate segments for BERT input."""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == 102:
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def register_hooks(model):
    """Register hooks to capture weights, biases, and gradients."""
    def hook_fn(module, input, output, name):
        if hasattr(module, 'weight'):
            wandb.log({f"{name}.weight": wandb.Histogram(module.weight.data.cpu().numpy())})
        if hasattr(module, 'bias') and module.bias is not None:
            wandb.log({f"{name}.bias": wandb.Histogram(module.bias.data.cpu().numpy())})

    def hook_fn_grad(module, grad_input, grad_output, name):
        wandb.log({f"{name}.grad": wandb.Histogram(grad_input[0].cpu().numpy())})

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
            module.register_backward_hook(lambda module, grad_input, grad_output, name=name: hook_fn_grad(module, grad_input, grad_output, name))

def process_batch(batch, batch_number, tokenizer, model):
    """Process a batch through BERT."""
    logging.info(f"Processing batch {batch_number}...")
    pair = batch['Title1'] + batch['Description1'] + " [SEP] " + batch['Title2'] + batch['Description2']
    tokenized = pair.apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LENGTH))
    
    padded = np.array([i + [0]*(MAX_SEQ_LENGTH-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    input_segments = np.array([_get_segments(token, MAX_SEQ_LENGTH) for token in tokenized.values])
    token_type_ids = torch.tensor(input_segments)
    
    last_hidden_states = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    dummy_target = torch.randn_like(last_hidden_states[0][:, 0, :])
    loss = nn.MSELoss()(last_hidden_states[0][:, 0, :], dummy_target)
    loss.backward()
    
    features = last_hidden_states[0][:, 0, :].detach().numpy()
    return features

def clean_label_noise(features, labels):
    """Clean label noise using confidence learning."""
    logging.info("Cleaning label noise using confidence learning...")
    cl = classification.CleanLearning()
    features_cleaned, labels_cleaned = cl.fit_transform(features, labels)
    return features_cleaned, labels_cleaned

def handle_class_imbalance(features, labels):
    """Handle class imbalance using SMOTE."""
    logging.info("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    return features_balanced, labels_balanced

def train_mlp_classifier(train_features, train_labels):
    """Train MLP Classifier with RandomizedSearchCV."""
    logging.info("Training MLP Classifier with RandomizedSearchCV...")
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': uniform(0.0001, 0.05),
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': randint(100, 300)
    }
    
    mlp = MLPClassifier(random_state=42)
    random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=1)
    random_search.fit(train_features, train_labels)
    
    wandb.config.update(random_search.best_params_, allow_val_change=True)
    logging.info("Best parameters found:\n" + str(random_search.best_params_))
    
    return random_search

def evaluate_model(model, test_features, test_labels):
    """Evaluate the model and log metrics."""
    logging.info("Evaluating the model...")
    y_pred = model.predict(test_features)
    print('Results on the test set:')
    print(classification_report(test_labels, y_pred))
    print(confusion_matrix(test_labels, y_pred))
    
    wandb.log({
        "classification_report": classification_report(test_labels, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(test_labels, y_pred)
    })

def main(args):
    initialize_wandb()
    
    df1, df2 = load_datasets()
    df1, df2 = remove_stop_words(df1, df2)
    batches, test_batches = create_batches(df1, df2)
    
    logging.info("Loading pre-trained BERT model and tokenizer...")
    tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')
    model = ppb.BertModel.from_pretrained('bert-base-uncased')
    
    register_hooks(model)
    model.train()
    
    logging.info("Processing batches through BERT...")
    features_list = [process_batch(batch, i+1, tokenizer, model) for i, batch in enumerate(batches + test_batches)]
    features = np.concatenate(features_list)
    
    logging.info("Preparing data for classification...")
    Total = pd.concat(batches + test_batches, ignore_index=True)
    labels = Total['Label']
    
    train_features = features[0:6792]
    train_labels = labels[0:6792]
    test_features = features[6792:]
    test_labels = labels[6792:]
    
    # Handle label noise if specified
    if args.label_noise:
        train_features, train_labels = clean_label_noise(train_features, train_labels)
    
    # Handle class imbalance if specified
    if args.class_imbalance:
        train_features, train_labels = handle_class_imbalance(train_features, train_labels)
    
    model = train_mlp_classifier(train_features, train_labels)
    evaluate_model(model, test_features, test_labels)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Duplicate Bug Report Detection")
    parser.add_argument("-label-noise", action="store_true", help="Clean label noise using confidence learning")
    parser.add_argument("-class-imbalance", action="store_true", help="Handle class imbalance using SMOTE")
    args = parser.parse_args()
    
    main(args)
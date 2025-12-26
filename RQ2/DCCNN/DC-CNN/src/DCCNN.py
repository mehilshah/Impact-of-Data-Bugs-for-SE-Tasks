import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
from imblearn.over_sampling import SMOTE
from modules import represent_training_pairs
wandb.init(project="case-study-duplicate-bug-report-detection")
SEED = 42
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def set_global_determinism(seed=42):
    set_seeds(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(SEED)

def get_strategy(gpu_fraction=1.0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_fraction)])
        except RuntimeError as e:
            print(e)

def DCCNN_Model(input_shape):
    intermediate_outputs = []
    X_input = Input(shape=input_shape)
    
    X_1 = Conv2D(100, kernel_size=(1,20), strides=(1,1), activation='relu')(X_input)
    X_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_1)
    X_1 = Reshape((300,100,1))(X_1)
    intermediate_outputs.append(X_1)

    X_1_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1), activation='relu')(X_1)
    X_1_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_1)
    X_1_1 = MaxPooling2D(pool_size=(300,1), padding='valid')(X_1_1)
    X_1_1 = Flatten()(X_1_1)
    intermediate_outputs.append(X_1_1)
    X_1_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1), activation='relu')(X_1)
    X_1_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_2)
    X_1_2 = MaxPooling2D(pool_size=(299,1), padding='valid')(X_1_2)
    X_1_2 = Flatten()(X_1_2)
    intermediate_outputs.append(X_1_2)
    X_1_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1), activation='relu')(X_1)
    X_1_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_1_3)
    X_1_3 = MaxPooling2D(pool_size=(298,1), padding='valid')(X_1_3)
    X_1_3 = Flatten()(X_1_3)
    intermediate_outputs.append(X_1_3)
    
    X_1 = tf.keras.layers.Concatenate(axis=-1)([X_1_1,X_1_2])
    X_1 = tf.keras.layers.Concatenate(axis=-1)([X_1,X_1_3])
    intermediate_outputs.append(X_1)

    X_2 = Conv2D(100, kernel_size=(2,20), strides=(1,1), activation='relu')(X_input)
    X_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_2)
    X_2 = Reshape((299,100,1))(X_2)
    intermediate_outputs.append(X_2)

    X_2_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1), activation='relu')(X_2)
    X_2_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_1)
    X_2_1 = MaxPooling2D(pool_size=(299,1), padding='valid')(X_2_1)
    X_2_1 = Flatten()(X_2_1)
    intermediate_outputs.append(X_2_1)
    X_2_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1), activation='relu')(X_2)
    X_2_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_2)
    X_2_2 = MaxPooling2D(pool_size=(298,1), padding='valid')(X_2_2)
    X_2_2 = Flatten()(X_2_2)
    intermediate_outputs.append(X_2_2)
    X_2_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1), activation='relu')(X_2)
    X_2_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_2_3)
    X_2_3 = MaxPooling2D(pool_size=(297,1),padding='valid')(X_2_3)
    X_2_3 = Flatten()(X_2_3)
    intermediate_outputs.append(X_2_3)
    
    X_2 = tf.keras.layers.Concatenate(axis=-1)([X_2_1,X_2_2])
    X_2 = tf.keras.layers.Concatenate(axis=-1)([X_2,X_2_3])
    intermediate_outputs.append(X_2)
    
    X_3 = Conv2D(100, kernel_size=(3,20), strides=(1,1),activation='relu')(X_input)
    X_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_3)
    X_3 = Reshape((298,100,1))(X_3)
    intermediate_outputs.append(X_3)
    
    X_3_1 = Conv2D(200, kernel_size=(1,100), strides=(1,1),activation='relu')(X_3)
    X_3_1 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_1)
    X_3_1 = MaxPooling2D(pool_size=(298,1),padding='valid')(X_3_1)
    X_3_1 = Flatten()(X_3_1)
    intermediate_outputs.append(X_3_1)
    X_3_2 = Conv2D(200, kernel_size=(2,100), strides=(1,1),activation='relu')(X_3)
    X_3_2 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_2)
    X_3_2 = MaxPooling2D(pool_size=(297,1),padding='valid')(X_3_2)
    X_3_2 = Flatten()(X_3_2)
    X_3_3 = Conv2D(200, kernel_size=(3,100), strides=(1,1),activation='relu')(X_3)
    X_3_3 = tf.keras.layers.BatchNormalization(axis=-1)(X_3_3)
    X_3_3 = MaxPooling2D(pool_size=(296,1),padding='valid')(X_3_3)
    X_3_3 = Flatten()(X_3_3)
    intermediate_outputs.append(X_3_3)
    
    X_3 = tf.keras.layers.Concatenate(axis=-1)([X_3_1,X_3_2])
    X_3 = tf.keras.layers.Concatenate(axis=-1)([X_3,X_3_3])
    intermediate_outputs.append(X_3)
    
    X = tf.keras.layers.Concatenate(axis=-1)([X_1,X_2])
    X = tf.keras.layers.Concatenate(axis=-1)([X,X_3])
    intermediate_outputs.append(X)

    X = Dropout(0.6)(X)
    X = Dense(300, activation='relu')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    intermediate_outputs.append(X)
    
    X = Dropout(0.4)(X)
    X = Dense(100, activation='relu')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    intermediate_outputs.append(X)

    X = Dropout(0.4)(X)
    Y = Dense(1, activation='sigmoid')(X)
    intermediate_outputs.append(Y)
    
    model = Model(inputs=X_input, outputs=intermediate_outputs, name='CNN_Model')
    
    print(model.summary())
    
    return model

class IntermediateOutputCallback(Callback):
    def __init__(self, model, data, labels, log_interval=1):
        self.model = model
        self.data = data
        self.labels = labels
        self.log_interval = log_interval
        self.intermediate_outputs = {layer.name: [] for layer in model.layers}
        self.layer_output_models = Model(inputs=model.input, outputs=[layer.output for layer in model.layers])

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            outputs = self.layer_output_models(self.data)
            for layer_name, output in zip(self.intermediate_outputs.keys(), outputs):
                self.intermediate_outputs[layer_name].append(output)

    def on_train_end(self, logs=None):
        for layer_name, outputs in self.intermediate_outputs.items():
            # Aggregate outputs over epochs
            intermediate_output = np.concatenate(outputs, axis=0)
            # Perform t-SNE on the outputs
            tsne = TSNE(n_components=2, random_state=SEED)
            tsne_results = tsne.fit_transform(intermediate_output)

            # Plotting the t-SNE results
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.labels, cmap='viridis')
            plt.colorbar()
            plt.title(f't-SNE of {layer_name} Outputs')
            plt.show()

# write a method for addressing label noise using confidence learning in cleanlab
def address_label_noise(data_train, label_train):
    temp_model = DCCNN_Model((300, 20, 2))
    temp_model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    temp_model.fit(
        x=data_train,
        y=label_train,
        batch_size=64,
        epochs=100,
        validation_split=0.2,
        shuffle=True
    )
    
    probabilities = temp_model.predict(data_train).flatten()
    confidence_scores = np.abs(probabilities - 0.5)
    high_confidence_threshold = np.percentile(confidence_scores, 75)
    high_confidence_indices = np.where(confidence_scores >= high_confidence_threshold)[0]
    data_high_confidence = data_train[high_confidence_indices]
    labels_high_confidence = label_train[high_confidence_indices]
    return data_high_confidence, labels_high_confidence, np.ones(len(high_confidence_indices))

def address_class_imbalance(data_train, label_train, imbalance_strategy='smote'):
    """
    Address class imbalance in the training data using SMOTE.

    Parameters:
        data_train (numpy.ndarray): Training data features.
        label_train (numpy.ndarray): Training data labels.
        imbalance_strategy (str): Strategy for handling imbalance. Supports 'smote'.

    Returns:
        numpy.ndarray: Resampled training data features.
        numpy.ndarray: Resampled training data labels.
    """
    if imbalance_strategy.lower() == 'smote':
        smote = SMOTE(random_state=42)
        data_train_resampled, label_train_resampled = smote.fit_resample(data_train, label_train)
        return data_train_resampled, label_train_resampled
    else:
        raise ValueError("Unsupported imbalance strategy: {}".format(imbalance_strategy))

    return data_train, label_train  # Return original data if no strategy is applied

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name...')
    parser.add_argument('--project', help='project name', required=True)
    parser.add_argument('--label-noise', help='Address label noise while training the model', required=False)
    parser.add_argument('--class-imbalance', help='Address class imbalance while training the model', required=False)

    args = parser.parse_args()

    matrix_data_path = Path('../data/matrix/{}'.format(args.project))
    model_path = Path('../model/{}'.format(args.project))
    model_path.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    print('It started at: %s' % start_time)
    
    # Assume get_strategy function is defined elsewhere in the code
    get_strategy(1.0)  # using 100% of total GPU Memory

    sabd_data_path = '../../SABD/dataset/{}/'.format(args.project)
    hindbr_train_pairs = '../../HINDBR/data/model_training/{}_training_pairs.txt'.format(args.project)
    matrix_data_path = '../data/matrix/{}/'.format(args.project)
    
    data_train, label_train = represent_training_pairs(
        train_pairs=hindbr_train_pairs, 
        database_path=sabd_data_path + '{}.json'.format(args.project), \
        matrix_data_path=matrix_data_path
    )

    sample_weights = np.ones(len(label_train))  # Default weights

    if args.label_noise:
        data_train, labels_train, clean_flags = address_label_noise(data_train, label_train)
    if args.class_imbalance:
        data_train, label_train = address_class_imbalance(data_train, label_train, args.class_imbalance)
    
    label_train = label_train.astype(int)

    dccnnModel = DCCNN_Model((300, 20, 2))
    intermediate_callback = IntermediateOutputCallback(model=dccnnModel,  data=data_train, labels=label_train, log_interval=1)
    
    dccnnModel.compile(
        optimizer=keras.optimizers.Adam(lr=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    wandb.init(project=args.project)

    dccnnModel.fit(
        x=data_train, 
        y=label_train,
        sample_weight=sample_weights,
        batch_size=64, 
        epochs=100,
        validation_split=0.2, 
        callbacks=[wandb.keras.WandbCallback(), intermediate_callback],
        shuffle=True
    )
    
    end_time = datetime.now()
    print('It ended at: %s' % end_time)
    
    for i in range(1, 20):
        if not os.path.exists(model_path / 'dccnn_{}.h5'.format(i)):
            model_save_name = model_path / 'dccnn_{}.h5'.format(i)
            break

    dccnnModel.save(model_save_name)
    wandb.finish()
    tf.keras.backend.clear_session()

import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Reproductibility
from numpy.random import seed
seed(1002)
import tensorflow
tensorflow.random.set_seed(1002)


def keras_model_fn():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim = 18, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='df_train.csv')
    parser.add_argument('--test-file', type=str, default='df_test.csv')

    args, _ = parser.parse_known_args()
    print(args.train)
    SEED = 42
    print('reading data')
    df_train = pd.read_csv(os.path.join(args.train, args.train_file))
    df_test = pd.read_csv(os.path.join(args.test, args.test_file))
    
    # X and Y
    X_train = df_train.iloc[:, 1:20].values
    y_train = df_train.iloc[:,0].values

    model = keras_model_fn()
    model.fit(X_train, y_train, epochs = 20, batch_size = 10)
    
    model_dir = os.environ.get('SM_MODEL_DIR')
    
    print(f'model_dir {model_dir}')
    # save Keras model for Tensorflow Serving
    version='0000'
    tensorflow.saved_model.save(model, os.path.join(model_dir, version))

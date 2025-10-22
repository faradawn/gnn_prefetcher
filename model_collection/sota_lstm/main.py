#!/usr/bin/env python3
import csv
import sys
import pandas as pd
import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import glob
import csv
import sys
import os
import math
import time
import tensorflow as tf
import pickle

# ingore gpu
tf.config.set_visible_devices([], 'GPU')

import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Flatten, Embedding, LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from sklearn import metrics
from tensorflow.keras.layers import Dense, Activation, TimeDistributed, Dense
from keras.callbacks import ModelCheckpoint
from numpy import insert
from sklearn import preprocessing
from collections import Counter
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import csv
import sys
import pandas as pd
import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import glob
import csv
import sys
import os
import math
import tensorflow as tf

# ingore gpu
tf.config.set_visible_devices([], 'GPU')

import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Flatten, Embedding, LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from sklearn import metrics
from tensorflow.keras.layers import Dense, Activation, TimeDistributed, Dense
from keras.callbacks import ModelCheckpoint
from numpy import insert
from sklearn import preprocessing
from collections import Counter
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def build_delta_map(train_trace, top_num=1000):
    x = Counter(train_trace['KByteOffset_Delta'])
    vals = x.most_common(top_num)
    top_deltas = {}
    rev_map = {}
    for i, tup in enumerate(vals):
        top_deltas[tup[0]] = i
        rev_map[i] = tup[0] # i -> raw delta
    
    forward_map = {}
    count = 0
    while (count < len(train_trace)):
        x = train_trace['KByteOffset_Delta'].iloc[count]
        if x in top_deltas:
            forward_map[x] = top_deltas[x]
        count += 1
    return forward_map, rev_map


# === 5. train
num_epochs = 5
batch_size = 32
delta_map = {}
size_map = {}
rev_delta_map = {}
rev_size_map = {}
model = None

def set_model_globally(model_):
    global model
    model = model_

def conver_delta_class_to_value(delta_class):
    global rev_delta_map
    if delta_class == 1000:
        return None
    else:
        return rev_delta_map[delta_class]

def convert_class_to_values(delta_class, size_class):
    global rev_delta_map, rev_size_map
    delta = conver_delta_class_to_value(delta_class)
    if delta == None:
        # no prefetching
        delta, size = None, None
    else:
        delta = rev_delta_map[delta_class]
        size = rev_size_map[size_class]
    return delta, size

def build_input_features(arr_raw_delta, arr_raw_size):
    global delta_map, size_map
    arr_delta_class = []
    arr_size_class = []
    for idx in range(len(arr_raw_delta)):
        delta = arr_raw_delta[idx]
        size = arr_raw_size[idx]
        # if can't find it in the mapping, we will use the last class
        delta_class = delta_map.get(delta, len(delta_map) - 1)
        size_class = size_map.get(size, len(size_map) - 1)
        arr_delta_class.append(delta_class)
        arr_size_class.append(size_class)
    # arr_delta_class = [   7,    0, 1000,    0, 1000, 1000,   31,   20,   58,   18,   63,
    #     1000,   12,    0,  128,   18,    0,   35,    7,    0, 1000, 1000,
    #     1000,    0, 1000, 1000,   17, 1000,    6,   10,   38,   17]
    # arr_size_class = [1, 2, 0, 2, 0, 1, 1, 2, 0, 5, 2, 3, 1, 0, 0, 1, 0, 0, 1, 2, 0, 1,
    #     1, 0, 0, 0, 1, 2, 1, 0, 0, 3]
    input_features = [np.array([arr_delta_class]), np.array([arr_size_class])]
    return input_features

def get_list_lba(last_lba, arr_delta, arr_size):
    arr_lba = []
    prev_lba = last_lba
    for idx, delta in enumerate(arr_delta):
        size = arr_size[idx]
        if delta != None:
            prev_lba -= delta
            arr_lba.append([prev_lba, size])
    return arr_lba

def predict_delta_and_size(arr_raw_delta, arr_raw_size):
    global model
    input_features = build_input_features(arr_raw_delta, arr_raw_size)
    pred_delta, pred_size = model.predict(input_features, verbose = 0)
    pred_delta, pred_size = pred_delta[0], pred_size[0] # select the first prediction
    arr_delta, arr_size = [], []
    for idx in range(len(pred_delta)): # 32 LBA delta predictions
        # find out the predicted delta and size 
        delta_class = np.argmax(pred_delta[idx])
        size_class = np.argmax(pred_size[idx])
        delta, size = convert_class_to_values(delta_class, size_class)
        arr_delta.append(delta)
        arr_size.append(size)
        # print(idx,"curr_offset", arr_raw_delta[idx], "\tdelta", delta, "size", size)
    return arr_delta, arr_size 

def train_model(df_raw):
    # the unit here is KB (not byte)
    global num_epochs, batch_size, rev_delta_map, rev_size_map, delta_map, size_map

    df = pd.DataFrame(df_raw)
    print("Training model")

    # === 1. build delta map
    df['KByteOffset_Delta'] = df['offset'] - df['offset'].shift(-1)
    df = df.drop(df.index[-1])
    df['KByteOffset_Delta'] = df['KByteOffset_Delta'].fillna(0)

    delta_map, rev_delta_map = build_delta_map(df)
    df['KByteOffset_Delta_Class_1001'] = df['KByteOffset_Delta'].map(lambda x: delta_map.get(x, 1000))

    # === 2. build size map
    df['IOSize_log'] = np.log2(df['size'])      # np.log2(16) = 4.0
    df['IOSize_log_roundoff']= round(df['IOSize_log'])
    a = df['IOSize_log_roundoff'].unique().tolist()
    size_map = {}
    rev_size_map = {}
    for i,id in enumerate(a):
        rev_size_map[i] = id
        size_map[id] = i 
    df['Size_Class'] = df['IOSize_log_roundoff'].map(lambda x: size_map[x])

    # print("# Delta classes unique", df['KByteOffset_Delta_Class_1001'].nunique())
    # print("# Size classes unique", df['Size_Class'].nunique())
    # print("# Size log2 unique", df['IOSize_log_roundoff'].nunique())
    # print("# Raw size unique", df['size'].nunique())
    # print("size_map", size_map)
    print("rev_size_map", rev_size_map)

    # save the four maps
    # with open('maps.pkl', 'wb') as f:
    #     pickle.dump((delta_map, rev_delta_map, size_map, rev_size_map), f)

    # === 3. Split to train, validate and test
    print("\n=== Split training testing")
    training_pt_1 = math.floor((len(df)*0.75)) 

    delta_lba_train =df[:training_pt_1]['KByteOffset_Delta_Class_1001'].tolist()
    delta_lba_test = df[training_pt_1+1:]['KByteOffset_Delta_Class_1001'].tolist()
    raw_lba_test = df[training_pt_1+1:]['offset'].tolist()
    raw_delta_test = df[training_pt_1+1:]['KByteOffset_Delta'].tolist()
    # print("raw_lba_test", len(raw_lba_test), raw_lba_test[0:32])
    # print("raw_delta_test", len(raw_delta_test), raw_delta_test[0:32])
    # print("delta_lba_test", len(delta_lba_test), delta_lba_test[0:32])
    size_train = df[:training_pt_1]['Size_Class'].tolist()
    size_test = df[training_pt_1+1:]['Size_Class'].tolist()

    delta_lba_train= np.array(delta_lba_train).reshape(-1,1) # each row only one element
    delta_lba_test= np.array(delta_lba_test).reshape(-1,1)
    size_train= np.array(size_train).reshape(-1,1)
    size_test= np.array(size_test).reshape(-1,1)

    # Each x is [1, ..., 32] and y is [33, ..., 65] (using 32 words to predict the next 32 words)
    def create_dataset2(dataset, window_size):
        dataX, dataY = [], []
        for i in range(len(dataset) - 2 * window_size):
            a = dataset[i:(i + window_size), 0] # a is [1,2,3,4 ...], next [2,3,4,5 ...]
            dataX.append(a)
            b = dataset[(i + window_size):(i + 2* window_size), 0]
            dataY.append(b) # 
        return np.array(dataX), np.array(dataY)

    lstm_num_timesteps = 32
    
    # All of data here are the classes, not the raw values
    X_train_delta_lba, y_train_delta_lba = create_dataset2(delta_lba_train, lstm_num_timesteps)
    X_test_delta_lba, y_test_delta_lba = create_dataset2(delta_lba_test, lstm_num_timesteps)
    # print("delta_lba_test", len(delta_lba_test), delta_lba_test[0:32].tolist())
    # print("X_test_delta_lba", len(X_test_delta_lba), X_test_delta_lba[0])
    X_train_size, y_train_size = create_dataset2(size_train, lstm_num_timesteps)
    X_test_size, y_test_size = create_dataset2(size_test, lstm_num_timesteps)

    lstm_num_features = 1
    lstm_predict_sequences = True
    lstm_num_predictions = 32

    # X_train = np.reshape(X_train, (X_train.shape[0], lstm_num_timesteps, lstm_num_features))
    # X_test = np.reshape(X_test, (X_test.shape[0], lstm_num_timesteps, lstm_num_features))
        
    y_train_delta_lba = np.reshape(y_train_delta_lba, (y_train_delta_lba.shape[0], lstm_num_predictions, lstm_num_features))
    y_test_delta_lba = np.reshape(y_test_delta_lba, (y_test_delta_lba.shape[0], lstm_num_predictions, lstm_num_features))
    y_train_size = np.reshape(y_train_size, (y_train_size.shape[0], lstm_num_predictions, lstm_num_features))
    y_test_size = np.reshape(y_test_size, (y_test_size.shape[0], lstm_num_predictions, lstm_num_features))                        

    # print(f'X_train_lba {X_train_delta_lba.shape}, X_test_lba {X_test_delta_lba.shape}, X_train_size {X_train_size.shape}, X_test_size {X_test_size.shape}')
    # print(f'y_train_lba {y_train_delta_lba.shape}, y_test_lba {y_test_delta_lba.shape}, y_train_size {y_train_size.shape}, y_test_size {y_test_size.shape}')

    hidden_size = 150 # original 1500, acc not increasing with more hidden units. 39$ vs 35%

    # === 4. building model
    maxlen= 32

    # # define two sets of inputs
    # inputA = Input(shape=(32,))
    # inputB = Input(shape=(32,))
    # # inputA = Sequential()
    # # inputB = Sequential()
    vocabulary_1 = df['KByteOffset_Delta_Class_1001'].nunique()
    vocabulary_2 = df['Size_Class'].nunique()

    # print("vocab 1 Delta_Class", vocabulary_1)
    # print("vocab 2 Size_class", vocabulary_2)

    # input=Input(shape=(no_docs,maxlen),dtype='float64')
    inputA=Input(shape=(maxlen,),dtype='float64')  
    inputB=Input(shape=(maxlen,),dtype='float64') 


    # the first branch operates on the first input
    x = Embedding(input_dim=vocabulary_1,output_dim=hidden_size,input_length=maxlen)(inputA)
    x = Model(inputs=inputA, outputs=x)

    # # the second branch opreates on the second input
    y = Embedding(input_dim=vocabulary_2,output_dim=hidden_size,input_length=maxlen)(inputB)
    y = Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = tf.keras.layers.concatenate([x.output, y.output])

    lstm1 = LSTM(hidden_size,return_sequences=True)(combined)
    lstm2 = LSTM(hidden_size, return_sequences=True)(lstm1)

    # create classification output
    offset = TimeDistributed(Dense(units=vocabulary_1, activation='softmax'), name='offset')(lstm2)
    iosize = TimeDistributed(Dense(units=vocabulary_2, activation='softmax'), name='iosize')(lstm2)

    model =Model([inputA,inputB],[offset,iosize]) # combining all into a Keras model

    model.compile(optimizer='rmsprop',
                loss={'offset': 'sparse_categorical_crossentropy', 'iosize': 'sparse_categorical_crossentropy'},
                loss_weights={'offset': 2., 'iosize': 1.5},
                metrics={ 'offset': 'categorical_accuracy', 'iosize': 'categorical_accuracy'})
    # model.summary()

    print('Train...')
    start_time = time.time()
    monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    valid = [X_test_delta_lba,X_test_size],[y_test_delta_lba,y_test_size]

    model.fit([X_train_delta_lba,X_train_size],[y_train_delta_lba,y_train_size],
            verbose=1, epochs=num_epochs, callbacks=[monitor], batch_size=batch_size, validation_data=valid)

    # No need to save the model to a file
    # model.load_weights('best_weights_src1.hdf5') # load weights from best model
    # model_path = f'lstm_hidden_{hidden_size}_epochs_{num_epochs}.keras'
    # model.save(model_path)
    # print(f"Model saved to {model_path}")
    # print('Done, elapsed', time.time() - start_time)


    # === 6. Inference
    new_model = model # Just use this model; no need to load from file
    # new_model = keras.models.load_model(model_path)
    pred1,pred2 = new_model.predict([X_test_delta_lba,X_test_size],verbose =1 )
    # print("X_test_lba.shape", X_test_delta_lba.shape, X_test_delta_lba[0])
    # print("X_test_size.shape", X_test_size.shape, X_test_size[0])
    # print("pred1 pred2 shape", pred1.shape, pred2.shape)

    # Check accuracy
    pred_1 = pred1[:,0,:] # select the first prediction (1959, 1); Q: But WHY?? 
    pred_2 = pred2[:,0,:] # select the first prediction (1959, 1)
    pred_1 = np.argmax(pred_1, axis=1)
    pred_2 = np.argmax(pred_2, axis=1)

    lba_test_final = delta_lba_test[-(len(pred1)):]
    lba_size_final = size_test[-(len(pred1)):]
    lba_accuracy = accuracy_score(lba_test_final, pred_1)
    size_accuracy = accuracy_score(lba_size_final, pred_2)
    print("lba_accuracy", lba_accuracy)
    print("size_accuracy", size_accuracy)

    # Test on 1 inference
    print((X_test_delta_lba[:1]).shape)
    arr_delta_class = X_test_delta_lba[0]
    arr_size_class = X_test_size[0]
    input_features = [np.array([arr_delta_class]), np.array([arr_size_class])]
    print("sample input_features", input_features)
    pred_delta, pred_size = new_model.predict(input_features,verbose = 0)
    pred_delta, pred_size = pred_delta[0], pred_size[0] # select the first prediction
    # print("rev_delta_map", rev_delta_map)
    for idx in range(len(pred_delta)): # 32 LBA delta predictions
        # find out the predicted delta and size 
        delta_class = np.argmax(pred_delta[idx])
        size_class = np.argmax(pred_size[idx])
        delta, size = convert_class_to_values(delta_class, size_class)
        # print(idx,"curr_offset", raw_lba_test[idx], "\tdelta", delta, "size", size)

    # print("pred_delta class", pred_delta)
    # print("pred_size class", pred_size)
    # delta_lba_test 1111 [7, 0, 1000, 0, 1000, 1000, 31, 20, 58, 18, 63, 1000, 12, 0, 128, 18, 0, 35, 7, 0, 1000, 1000, 1000, 0, 1000, 1000, 17, 1000, 6, 10, 38, 17]
    return new_model

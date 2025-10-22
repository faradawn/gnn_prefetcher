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
    x = Counter(train_trace['ByteOffset_Delta'])
    vals = x.most_common(top_num)
    top_deltas = {}
    rev_map = {}
    for i, tup in enumerate(vals):
        top_deltas[tup[0]] = i
        rev_map[i] = tup[0] # i -> raw delta
    
    forward_map = {}
    count = 0
    while (count < len(train_trace)):
        x = train_trace['ByteOffset_Delta'].iloc[count]
        if x in top_deltas:
            forward_map[x] = top_deltas[x]
        count += 1
    return forward_map, rev_map

# === 1. build delta map

input_file = '../../dataset/iotrace/alibaba.cut.per_10k.most_size_thpt_iops_rand.719/read_io.trace'
df = pd.read_csv(input_file, sep=',')
df['ByteOffset_Delta'] = df['offset'] - df['offset'].shift(-1)
df = df.drop(df.index[-1])
df['ByteOffset_Delta'] = df['ByteOffset_Delta'].fillna(0)

delta_map, rev_delta_map = build_delta_map(df)
df['ByteOffset_Delta_Class_1001'] = df['ByteOffset_Delta'].map(lambda x: delta_map.get(x, 1000))

# === 2. build size map

df['IOSize_log'] = np.log2(df['size'])
df['IOSize_log_roundoff']= round(df['IOSize_log'])
a = df['IOSize_log_roundoff'].unique().tolist()
size_map = {}
rev_size_map = {}
for i,id in enumerate(a):
    rev_size_map[i] = id
    size_map[id] = i 
df['Size_Class'] = df['IOSize_log_roundoff'].map(lambda x: size_map[x])

print("Delta classes unique", df['ByteOffset_Delta_Class_1001'].nunique())

# save the four maps
import pickle
with open('maps.pkl', 'wb') as f:
    pickle.dump((delta_map, rev_delta_map, size_map, rev_size_map), f)


# === 3. Split to train, validate and test
print("\n=== Split training testing")
training_pt_1 = math.floor((len(df)*0.75)) 

lba_train =df[:training_pt_1]['ByteOffset_Delta_Class_1001'].tolist()
lba_test = df[training_pt_1+1:]['ByteOffset_Delta_Class_1001'].tolist()
size_train = df[:training_pt_1]['Size_Class'].tolist()
size_test = df[training_pt_1+1:]['Size_Class'].tolist()

lba_train= np.array(lba_train).reshape(-1,1) # each row only one element
lba_test= np.array(lba_test).reshape(-1,1)
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
    
X_train_lba, y_train_lba = create_dataset2(lba_train, lstm_num_timesteps)
X_test_lba, y_test_lba = create_dataset2(lba_test, lstm_num_timesteps)
X_train_size, y_train_size = create_dataset2(size_train, lstm_num_timesteps)
X_test_size, y_test_size = create_dataset2(size_test, lstm_num_timesteps)

lstm_num_features = 1
lstm_predict_sequences = True
lstm_num_predictions = 32

# X_train = np.reshape(X_train, (X_train.shape[0], lstm_num_timesteps, lstm_num_features))
# X_test = np.reshape(X_test, (X_test.shape[0], lstm_num_timesteps, lstm_num_features))
    
y_train_lba = np.reshape(y_train_lba, (y_train_lba.shape[0], lstm_num_predictions, lstm_num_features))
y_test_lba = np.reshape(y_test_lba, (y_test_lba.shape[0], lstm_num_predictions, lstm_num_features))
y_train_size = np.reshape(y_train_size, (y_train_size.shape[0], lstm_num_predictions, lstm_num_features))
y_test_size = np.reshape(y_test_size, (y_test_size.shape[0], lstm_num_predictions, lstm_num_features))                        

print(f'X_train_lba {X_train_lba.shape}, X_test_lba {X_test_lba.shape}, X_train_size {X_train_size.shape}, X_test_size {X_test_size.shape}')
print(f'y_train_lba {y_train_lba.shape}, y_test_lba {y_test_lba.shape}, y_train_size {y_train_size.shape}, y_test_size {y_test_size.shape}')

hidden_size = 150 # original 1500

# === 4. building model

maxlen= 32

# # define two sets of inputs
# inputA = Input(shape=(32,))
# inputB = Input(shape=(32,))
# # inputA = Sequential()
# # inputB = Sequential()
vocabulary_1 = df['ByteOffset_Delta_Class_1001'].nunique()
vocabulary_2 = df['Size_Class'].nunique()

print("vocab 1 Delta_Class", vocabulary_1)
print("vocab 2 Size_class", vocabulary_2)


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

# === 5. train

num_epochs = 5
batch_size = 32


print('Train...')
start_time = time.time()
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

valid = [X_test_lba,X_test_size],[y_test_lba,y_test_size]

model.fit([X_train_lba,X_train_size],[y_train_lba,y_train_size],
          verbose=1, epochs=num_epochs, callbacks=[monitor], batch_size=batch_size, validation_data=valid)

# model.load_weights('best_weights_src1.hdf5') # load weights from best model
model.save(f'lstm_hidden_{hidden_size}_epochs_{num_epochs}.keras')
print('Done, elapsed', time.time() - start_time)



# === 6. Inference
new_model = keras.models.load_model(f'lstm_hidden_{hidden_size}_epochs_{num_epochs}.keras')
pred1,pred2 = new_model.predict([X_test_lba,X_test_size],verbose =1 )
print("pred1 pred2 shape", pred1.shape, pred2.shape)

training_pt_1 = math.floor((len(df)*0.75)) 

lba_train =df[:training_pt_1]['ByteOffset_Delta_Class_1001'].tolist()
lba_test = df[training_pt_1+1:]['ByteOffset_Delta_Class_1001'].tolist()
size_train = df[:training_pt_1]['Size_Class'].tolist()
size_test = df[training_pt_1+1:]['Size_Class'].tolist()

# === Computer Accurarcy 

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

lba_test_final = lba_test[-(len(pred1)):]
lba_size_final = size_test[-(len(pred1)):]

max_lba_accuracy = 0
max_lba_accuracy_pos = 0

max_size_accuracy = 100000
max_size_accuracy_pos = 0

pred_1 = pred1[:,0,:] # select the first prediction (1959, 1)
pred_2 = pred2[:,0,:] # select the first prediction (1959, 1)
pred_1 = np.argmax(pred_1, axis=1)
pred_2 = np.argmax(pred_2, axis=1)

print("Trans pred_1 shape", len(pred_1), "test_final shape", len(lba_test_final))
# for i in range(len(pred_1)):
#     print(f"i {i}, pred_addr {pred_1[i]}, pred_size {pred_2[i]}")
                 

lba_accuracy = accuracy_score(lba_test_final, pred_1)
size_accuracy = accuracy_score(lba_size_final, pred_2)

print("IO Size Accuracy", str(size_accuracy))
print("Best LBA Delta Accuracy", str(lba_accuracy))

# cd '/home/cc/flashnet/model_collection/5_block_prefetching/simulate_sota_prefetcher/sota_lstm/'
# python train_lstm.py
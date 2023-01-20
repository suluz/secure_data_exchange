import pandas as pd
from os import path
import numpy as np
import tensorflow as tf
import statistics as stat
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import HyperParameters as hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import copy

# hyper-parameters
batch = 20 # batch size
batch = 20 # batch size
epochs = 100 # training epoch number
hidden_neuron_num = [100,100] # number of neurons in each hidden layer, you can add as many hidden layers as possible
hidden_neuron_num = [100,100] # number of neurons in each hidden layer, you can add as many hidden layers as possible
l_r = 0.1 # learning rate
l2_reg = 0.01 # l2 regulariser
exp_rounds = 1 # number of runs to produce average results
exp_rounds = 1 # number of runs to produce average results
train_acc_man = []
test_acc_man = []
timing_man = []
train_acc_keras = []
test_acc_keras = []
timing_keras = []

train_x_list = []
train_y_list = []

test_x_list = []
test_y_list = []

percentage_split_list = []


def read_data(df, label_col_index):
    # write you data reading function here
    # features should be min-max normalised
    # labels should be converted to one-hot encoded list
    # returned values are two numpy arrays

    # initialising the features and labels as the part of dataframes (series)
    arr = df.iloc[:,label_col_index].to_numpy()
    arr = df.iloc[:,label_col_index].to_numpy()
    df_labels = arr.reshape(-1,1)
    features = df.drop(df.columns[[label_col_index]], axis=1).to_numpy()
    features = df.drop(df.columns[[label_col_index]], axis=1).to_numpy()
    
    # scaler (min-max)
    # scaler = MinMaxScaler()
    # feats = scaler.fit_transform(features)  

    # no scaler
    feats = copy.deepcopy(features)

    # encoder (one-hot)
    encoder = OneHotEncoder(sparse=False) 
    # scaler (min-max)
    # scaler = MinMaxScaler()
    # feats = scaler.fit_transform(features)  

    # no scaler
    feats = copy.deepcopy(features)

    # encoder (one-hot)
    encoder = OneHotEncoder(sparse=False) 
    labels = encoder.fit_transform(df_labels)
    
    # return np.array(features), np.array(labels)
    return feats, labels
    # return np.array(features), np.array(labels)
    return feats, labels


def train_model(x_train, x_test, y_train, y_test):

    for i in range(exp_rounds):
        # print("***** Round %d *****" % i)
        # keras neural net
        tf.keras.backend.clear_session()
        layers = [
            # weights between input layer and the first hidden layer
            tf.keras.layers.Dense(
                hidden_neuron_num[0], # output size
                input_shape=(x_train.shape[1], ), # input size, just for the input layer 
                activation='sigmoid', 
                # kernel_initializer='zeros',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ]
        # weights between the first hidden layer and the second hidden layer
        # more layer can be added or deleted
        for j in range(1, 1, len(hidden_neuron_num)):
            layers.append(
                tf.keras.layers.Dense(
                    hidden_neuron_num[j], # output size
                    activation='sigmoid',
                    # kernel_initializer='zeros',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                    bias_initializer='zeros',
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
            )
        # add output layer
        layers.append(
            tf.keras.layers.Dense(
                y_train.shape[1], # output size
                activation='softmax', 
                # kernel_initializer='zeros',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        )
        model = tf.keras.Sequential(layers=layers)
        opt = tf.keras.optimizers.SGD(learning_rate=l_r, momentum=0.0)
        model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        model.fit(x_train, y_train, batch_size=batch, epochs=epochs, verbose=0)

        # stats
        train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
        train_acc_keras.append(train_accuracy*100)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        test_acc_keras.append(test_accuracy*100)

        # results
        print('Training accuracy:\n', train_accuracy*100)
        print('Test accuracy:\n', test_accuracy*100)


df_a = pd.read_csv('/content/secure_data_exchange/datasets/mixed_1010_abrupto_1.csv')
df_b = pd.read_csv('/content/secure_data_exchange/datasets/mixed_1010_abrupto_2.csv')

a_x, a_y = read_data(df_a,4)
b_x, b_y = read_data(df_b,4)
b_x_train, b_x_test, b_y_train, b_y_test = train_test_split(b_x, b_y, test_size=0.2, random_state=42)

# print(a_x.shape)
# print(b_x_train.shape)

split = 0.8
while split>=0.2:
    percentage_split_list.append(split)

    b_x_train_2, b_x_test_2, b_y_train_2, b_y_test_2 = train_test_split(b_x, b_y, train_size=split, random_state=42)
    
    # b_x_train_2 = b_x[:int(p)]
    # b_y_train_2 = b_y[:int(p)]

    # b_x_test_2 = b_x[int(p):, :]
    # b_y_test_2 = b_y[int(p):, :]

    c_x_train = np.append(a_x, b_x_train_2,axis=0)
    c_y_train = np.append(a_y, b_y_train_2,axis=0)

    train_x_list.append(c_x_train)
    train_y_list.append(c_y_train)
    test_x_list.append(b_x_test_2)
    test_y_list.append(b_y_test_2)

    split -= 0.2
    print(c_x_train.shape, c_y_train.shape)
    print(b_x_test_2.shape, b_x_test_2.shape)

train_model(a_x, b_x, a_y, b_y)

for i in range(len(percentage_split_list)):
    print(f"iteration {i+1} : A + {percentage_split_list[i]*100}% B\n\n")
    train_model(train_x_list[i], test_x_list[i], train_y_list[i], test_y_list[i])
    print("\n\n==================================================================================\n\n")
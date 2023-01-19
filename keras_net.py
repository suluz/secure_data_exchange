import numpy as np
import pandas as pd
from os import path
import tensorflow as tf
import statistics as stat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sys

# hyper-parameters
batch = 20 # batch size
epochs = 100 # training epoch number
hidden_neuron_num = [100,100] # number of neurons in each hidden layer, you can add as many hidden layers as possible
l_r = 0.1 # learning rate
l2_reg = 0.01 # l2 regulariser
exp_rounds = 1 # number of runs to produce average results
train_acc_man = []
test_acc_man = []
timing_man = []
train_acc_keras = []
test_acc_keras = []
timing_keras = []

# read data
# fill the read_data() func yourself
def read_data(df, label_col_index):
    # write you data reading function here
    # features should be min-max normalised
    # labels should be converted to one-hot encoded list
    # returned values are two numpy arrays

    # initialising the features and labels as the part of dataframes (series)
    arr = df.iloc[:,label_col_index].to_numpy()
    df_labels = arr.reshape(-1,1)
    features = df.drop(df.columns[[label_col_index]], axis=1).to_numpy()
    
    # initialising the scaler (min-max) and encoder (one-hot)
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse=False)

    feats = scaler.fit_transform(features)    
    labels = encoder.fit_transform(df_labels)
    
    # return np.array(features), np.array(labels)
    return feats, labels


def train_model(x_train, x_test, y_train, y_test):

    for i in range(exp_rounds):
        print("***** Round %d *****" % i)
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
        # print("Keras trained weights:\n", model.get_weights())

        # sys.exit()
        
    # print("\n"+"*"*20)
    # print("Hyper-parameters:")
    # print("learning rate: %.2f, l2 regularisor: %.2f, batch size: %d, epochs: %d" % (l_r, l2_reg, batch, epochs))
    # print("\n"+"*"*20)
    # print("%d Rounds Stats of Accuracy, Manual, Keras:" % exp_rounds)
    # print("Training Mean:", stat.mean(train_acc_keras))
    # print("Training Median:", stat.median(train_acc_keras))
    # print("Training Std Deviation:", stat.stdev(train_acc_keras))
    # print("Test Mean:", stat.mean(test_acc_keras))
    # print("Test Median:", stat.median(test_acc_keras))
    # print("Test Std Deviation:", stat.stdev(test_acc_keras))

# # test - read iris data
# df = pd.read_csv('./datasets/mixed_0101_abrupto.csv')
# x, y = read_data(df, 4)
# print(x)
# print(y)

# sys.exit()


df_a = pd.read_csv('./datasets/mixed_0101_abrupto.csv')
df_b = pd.read_csv('./datasets/mixed_1010_abrupto.csv')

a_x, a_y = read_data(df_a, 4)
a_x_train, a_x_test, a_y_train, a_y_test = train_test_split(a_x, a_y, test_size=0.2, random_state=42)

b_x, b_y = read_data(df_b, 4)
b_x_train, b_x_test, b_y_train, b_y_test = train_test_split(b_x, b_y, test_size=0.2, random_state=42)

df_c = pd.merge(df_a, df_b) # Adding two datasets (dataset A + dataset B)

c_x, c_y = read_data(df_c, 4)
c_x_train, c_x_test, c_y_train, c_y_test = train_test_split(c_x, c_y, test_size=0.2, random_state=42)

# # test
# print('y_train:\n', y_train)
train_model(a_x_train, b_x_test, a_y_train, b_y_test)
train_model(c_x_train, b_x_test, c_y_train, b_y_test)
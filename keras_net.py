import numpy as np
import pandas as pd
from os import path
import tensorflow as tf
import statistics as stat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# hyper-parameters
batch = 10
epochs = 100
hidden_neuron_num = 1
l_r = 0.1
l2_reg = 0.01
exp_rounds = 10
train_acc_man = []
test_acc_man = []
timing_man = []
train_acc_keras = []
test_acc_keras = []
timing_keras = []

# read data
# fill the read_data() func yourself
def read_data(df):
    # write you data reading function here
    # features should be min-max normalised
    # labels should be converted to one-hot encoded list
    # returned values are two numpy arrays

    # initialising the features and labels as the part of dataframes (series)
    df_features = df.iloc[:,:4]
    arr = df.iloc[:,4].to_numpy()
    df_labels = arr.reshape(-1,1)
    
    # initialising the scaler (min-max) and encoder (one-hot)
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse=False)

    features = scaler.fit_transform(df_features)    
    labels = encoder.fit_transform(df_labels)
    
    return np.array(features), np.array(labels)


def train_model(x_train, x_test, y_train, y_test):

    for i in range(exp_rounds):
        print("***** Round %d *****" % i)
        # keras neural net
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hidden_neuron_num, 
                input_shape=(x_train.shape[1], ), 
                activation='sigmoid', 
                kernel_initializer='zeros',
                # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            # tf.keras.layers.Dense(hidden_neuron_num, activation='sigmoid'),
            tf.keras.layers.Dense(
                y_train.shape[1], 
                activation='softmax', 
                kernel_initializer='zeros',
                # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ])
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
        print("Keras trained weights:\n", model.get_weights())

        # sys.exit()
        
    print("\n"+"*"*20)
    print("Hyper-parameters:")
    print("learning rate: %.2f, l2 regularisor: %.2f, batch size: %d, epochs: %d" % (l_r, l2_reg, batch, epochs))
    print("\n"+"*"*20)
    print("%d Rounds Stats of Accuracy, Manual, Keras:" % exp_rounds)
    print("Training Mean:", stat.mean(train_acc_keras))
    print("Training Median:", stat.median(train_acc_keras))
    print("Training Std Deviation:", stat.stdev(train_acc_keras))
    print("Test Mean:", stat.mean(test_acc_keras))
    print("Test Median:", stat.median(test_acc_keras))
    print("Test Std Deviation:", stat.stdev(test_acc_keras))

df_a = pd.read_csv('mixed_0101_abrupto.csv')
df_b = pd.read_csv('mixed_1010_abrupto.csv')

a_x, a_y = read_data(df_a)
a_x_train, a_x_test, a_y_train, a_y_test = train_test_split(a_x, a_y, test_size=0.2, random_state=42)

b_x, b_y = read_data(df_b)
b_x_train, b_x_test, b_y_train, b_y_test = train_test_split(b_x, b_y, test_size=0.2, random_state=42)

df_c = pd.merge(df_a, df_b) # Adding two datasets (dataset A + dataset B)

c_x, c_y = read_data(df_c)
c_x_train, c_x_test, c_y_train, c_y_test = train_test_split(c_x, c_y, test_size=0.2, random_state=42)

# # test
# print('y_train:\n', y_train)
train_model(a_x_train, b_x_test, a_y_train, b_y_test)
train_model(c_x_train, b_x_test, c_y_train, b_y_test)
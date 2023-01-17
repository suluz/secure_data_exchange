# pip install -q -U keras-tuner
# pip install -q tensorflow==2.3.0
import numpy as np
import pandas as pd
from os import path
import tensorflow as tf
import statistics as stat
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import HyperParameters as hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# hyper-parameters
batch = 10 # batch size
epochs = 100 # training epoch number
hidden_neuron_num = [1] # number of neurons in each hidden layer, you can add as many hidden layers as possible
l_r = 0.1 # learning rate
l2_reg = 0.01 # l2 regulariser
exp_rounds = 10 # number of runs to produce average results
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


def model_builder(hp):
            
    for i in range(exp_rounds):
        print("***** Round %d *****" % i)
        # keras neural net
        tf.keras.backend.clear_session()
        hp_units_1 = hp.Int('units1', min_value=32, max_value=512, step=32)
        hp_units_2 = hp.Int('units2', min_value=32, max_value=512, step=32)
        layers = [
            # weights between input layer and the first hidden layer
            tf.keras.layers.Dense(
                hp_units_1, # output size
                input_shape=(4, ), # input size, just for the input layer 
                activation='relu', 
                kernel_initializer='zeros',
                # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        ]
        # weights between the first hidden layer and the second hidden layer
        # more layer can be added or deleted
        for j in range(1, 1, len(hidden_neuron_num)):
            layers.append(
                tf.keras.layers.Dense(
                    hp_units_2, # output size
                    activation='relu',
                    kernel_initializer='zeros',
                    # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                    use_bias=True,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
            )
        # add output layer
        layers.append(
            tf.keras.layers.Dense(
                2, # output size
                activation='softmax', 
                kernel_initializer='zeros',
                # kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg))
        )

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

        model = tf.keras.Sequential(layers=layers)
        opt = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, momentum=0.0)

        

        model.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    return model

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

tuner_1 = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='model_1')


tuner_1.search(a_x_train , a_y_train, epochs=50, validation_split=0.2)

best_hps_a=tuner_1.get_best_hyperparameters(num_trials=1)[0]

model_1 = tuner_1.hypermodel.build(best_hps_a)
history = model_1.fit(a_x_train, a_y_train, epochs=20, validation_split=0.2)
val_acc_per_epoch_a = history.history['val_accuracy']
best_epoch_a = val_acc_per_epoch_a.index(max(val_acc_per_epoch_a)) + 1

print("\n Dataset = A, Tested on B\n")
print('Best epoch: %d' % (best_epoch_a,))
print('Best validation accuracy: %f' % (max(val_acc_per_epoch_a),))
print('Best hyperparameters: %s' % (best_hps_a,))
print("\n")

train_loss, train_accuracy = model_1.evaluate(a_x_train, a_y_train, verbose=0)
test_loss, test_accuracy = model_1.evaluate(b_x_test, b_y_test, verbose=0)
print('Training accuracy:\n', train_accuracy*100)
print('Test accuracy:\n', test_accuracy*100)

# Output of model 1:
# Training accuracy:
#  49.99687373638153
# Test accuracy:
#  50.0124990940094

tuner_2 = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='model_2')

tuner_2.search(c_x_train, c_y_train, epochs=50, validation_split=0.2)
best_hps_c=tuner_2.get_best_hyperparameters(num_trials=1)[0]

model_2 = tuner_2.hypermodel.build(best_hps_c)
history = model_2.fit(c_x_train, c_y_train, epochs=20, validation_split=0.2)

val_acc_per_epoch_c = history.history['val_accuracy']
best_epoch_c = val_acc_per_epoch_c.index(max(val_acc_per_epoch_c)) + 1

print("\n Dataset = A+B, Tested on B\n")
print('Best epoch: %d' % (best_epoch_c,))
print('Best validation accuracy: %f' % (max(val_acc_per_epoch_c),))
print('Best hyperparameters: %s' % (best_hps_c,))
print("\n")

train_loss_2, train_accuracy_2 = model_2.evaluate(c_x_train, c_y_train, verbose=0)
test_loss_2, test_accuracy_2 = model_2.evaluate(b_x_test, b_y_test, verbose=0)

print('Training accuracy:\n', train_accuracy_2*100)
print('Test accuracy:\n', test_accuracy_2*100)

# Output of model 2:
# Training accuracy:
#  50.07343888282776
# Test accuracy:
#  49.9875009059906
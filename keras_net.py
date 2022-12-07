import numpy as np
from os import path
import tensorflow as tf
import statistics as stat
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
x, y = read_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # test
# print('y_train:\n', y_train)

def read_data(self):
    # write you data reading function here
    # features should be min-max normalised
    # labels should be converted to one-hot encoded list
    # returned values are two numpy arrays
    return np.array(features), np.array(labels)

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
#  LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import tensorflow as tf

import local_info

# # gpu?
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
np.random.seed(7)

# ################################ SETTINGS ################################  #

# should we use the small dataset?
small_dataset = False
# are we saving the results?
saving_results = True
# are we storing part of the dataset for future use?
storing_small_dataset = False
# name of the file where saving results
output_name = "conv network test (only resp)"
output_name = output_name  \
              + "_" + str(datetime.date.today().day) \
              + "-" + str(datetime.date.today().month) \
              + "-" + str(datetime.date.today().year) \
              + "-" + str(datetime.datetime.now().hour) \
              + "h" + str("%02.f" % datetime.datetime.now().minute) \
              + ".csv"

# ################################ SETTINGS ################################  #

# load the dataset
print("Reading dataset...")
if small_dataset:
    print("Small dataset chosen")
    df_train = pd.read_csv(local_info.data_path + 'extract_train.csv')
    df_test = pd.read_csv(local_info.data_path + 'extract_test.csv')
else:
    print("Full dataset chosen")
    df_train = pd.read_csv(local_info.data_path+'train.csv')
    df_test = pd.read_csv(local_info.data_path+'test.csv')
dataset_train = df_train.values
dataset_train = dataset_train.astype('float32')
names = df_train.columns
dataset_test = df_train.values
dataset_test = dataset_test.astype('float32')
print("Done reading dataset! \n")

# stores an extract of 500 rows instead of 50k for debugging purposes
if storing_small_dataset:
    print("Storing extract...")
    os.chdir(local_info.data_path)
    temp_df1 = df_train[0:500]
    temp_df2 = df_test[0:500]
    temp_df1.to_csv('extract_train.csv', encoding='utf-8', index=False)
    temp_df2.to_csv('extract_test.csv', encoding='utf-8', index=False)
    print("Done storing extract! \n")

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.fit_transform(dataset_test)

# model
nb_epoch = 10
validation_share = 0.5
lr = 1

convolution = True
if convolution:
    model = Sequential()
    #model.add(Dense(units=200, activation='tanh', input_dim=3205))
    model.add(Conv1D(kernel_size=12, filters=32, input_shape=(400, 3)))
    model.add(Activation('relu'))
    model.add(Conv1D(kernel_size=5, filters=64, input_shape=(400, 3)))
    model.add(Activation('relu'))
    model.add(Conv1D(kernel_size=3, filters=128, input_shape=(400, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))
    sgd = SGD(lr=lr, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='mse', optimizer='rmsprop')
else:
    model = ensemble.GradientBoostingRegressor(n_estimators=100)

total_number = len(df_train)
valid_number = round(validation_share * total_number)
train_number = total_number - valid_number

if convolution:
    train_data_x = np.reshape(df_train[df_train.columns[2001:3201]][0:train_number].values, (train_number, 400, 3))
    train_data_y = df_train[df_train.columns[3206:3207]][0:train_number].values.ravel()

    valid_data_x = np.reshape(df_train[df_train.columns[2001:3201]][train_number:total_number].values, (valid_number, 400, 3))
    valid_data_y = df_train[df_train.columns[3206:3207]][train_number:total_number].values.ravel()

    # we remove first column which contains the user id (which is also the row index...)
    test_data_x = np.reshape(df_test[df_train.columns[2001:3201]].values, (total_number, 400, 3))

    # fitting the model
    model.fit(train_data_x, train_data_y, epochs=nb_epoch, validation_data=(valid_data_x, valid_data_y), batch_size=16)
else:
    train_data_x = df_train[df_train.columns[1:3206]][0:train_number].as_matrix()
    train_data_y = df_train[df_train.columns[3206:3207]][0:train_number].values.ravel()

    valid_data_x = df_train[df_train.columns[1:3206]][(train_number+1):total_number].as_matrix()
    valid_data_y = df_train[df_train.columns[3206:3207]][(train_number+1):total_number].values.ravel()

    # we remove first column which contains the user id (which is also the row index...)
    test_data_x = df_test[df_train.columns[1:3206]].as_matrix()
    # fitting the model
    model.fit(train_data_x, train_data_y)

print("Done fitting model!")

# make predictions
train_pred = model.predict(train_data_x)
valid_pred = model.predict(valid_data_x)
test_pred = model.predict(test_data_x)

# testPredict = model.predict(dataset_test)

train_score = math.sqrt(mean_squared_error(train_data_y, train_pred))
print('Train Score: %.2f RMSE' % train_score)

valid_score = math.sqrt(mean_squared_error(valid_data_y, valid_pred))
print('Validation Score: %.2f RMSE' % valid_score)

if saving_results:
    os.chdir(local_info.data_path)
    df_test_pred = pd.DataFrame(test_pred)
    df_test_pred.columns = ['power_increase']
    df_test_pred.index.names = ['index']
    df_test_pred.to_csv(output_name, encoding='utf-8')
    print("Stored dataset under the name '%s' at location '%s'" % (output_name+("-VS%.2f" % valid_score), os.getcwd()))

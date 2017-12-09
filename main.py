#  LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

import local_info

# gpu?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset (extract => work on small datasets for debugging
extract = True
print("Reading datasets...")
if extract:
    df_train = pd.read_csv(local_info.data_path + 'extract_train.csv')
    df_test = pd.read_csv(local_info.data_path + 'extract_test.csv')
else:
    df_train = pd.read_csv(local_info.data_path+'train.csv')
    df_test = pd.read_csv(local_info.data_path+'test.csv')
dataset_train = df_train.values
dataset_train = dataset_train.astype('float32')
names = df_train.columns
dataset_test = df_train.values
dataset_test = dataset_test.astype('float32')
print("Done reading datasets! \n")

# stores an extract of 50 rows instead of 50k for debugging purposes
storing = False
if storing:
    print("Storing extract...")
    os.chdir(local_info.data_path)
    temp_df1 = df_train[0:50]
    temp_df2 = df_test[0:50]
    temp_df1.to_csv('extract_train.csv', encoding='utf-8', index=False)
    temp_df2.to_csv('extract_test.csv', encoding='utf-8', index=False)
    print("Done storing extract! \n")

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.fit_transform(dataset_test)

# model
model = Sequential()
model.add(Conv1D(kernel_size=1, filters=512, input_shape=(3205, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='relu'))

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
nb_epoch = 15
validation_share = 0.5

total_number = len(df_train)
valid_number = round(validation_share * total_number)
train_number = total_number - valid_number

train_data_x = np.expand_dims(df_train[df_train.columns[1:3206]][0:train_number], axis=2)
train_data_y = df_train[df_train.columns[3206:3207]][0:train_number]

valid_data_x = np.expand_dims(df_train[df_train.columns[1:3206]][(train_number+1):total_number], axis=2)
valid_data_y = df_train[df_train.columns[3206:3207]][(train_number+1):total_number]

model.fit(train_data_x, train_data_y, epochs=nb_epoch, validation_data=(valid_data_x, valid_data_y), batch_size=16)

# make predictions
trainPredict = model.predict(dataset_train)
# testPredict = model.predict(dataset_test)

print(trainPredict)

# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
#
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset_train)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset_test)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset_train))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
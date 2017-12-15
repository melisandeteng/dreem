import numpy as np
import pandas as pd
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, Merge, MaxPooling1D
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import local_info
import utils


def convolve(small_dataset, saving_results, storing_small_dataset, output_name):
    apply_model(small_dataset, saving_results, storing_small_dataset, output_name, True, 0)


def gradient_boost(small_dataset, saving_results, storing_small_dataset, output_name, parameter):
    apply_model(small_dataset, saving_results, storing_small_dataset, output_name, False, parameter)


def apply_model(small_dataset, saving_results, storing_small_dataset, output_name, convolution, param):
    # load the dataset
    df_train, df_test = utils.load_dataset(small_dataset, storing_small_dataset)

    # normalize the dataset into [-1, +1]
    df_test = utils.normalize_dataframe(df_test)
    df_train = utils.normalize_dataframe(df_train)

    # parameters
    nb_epoch = 5 # maybe more?
    validation_share = 0.5

    # choose validation and training data from the training set
    total_number = len(df_train)
    valid_number = round(validation_share * total_number)
    train_number = total_number - valid_number

    # model
    if convolution:
        branch_eeg = Sequential()
        branch_eeg.add(Conv1D(kernel_size=20, filters=32, input_shape=(2000, 1), activation='relu'))
        branch_eeg.add(Dropout(0.5))
        branch_eeg.add(Conv1D(kernel_size=5, filters=64, activation='relu'))
        branch_eeg.add(MaxPooling1D(pool_size=3))
        branch_eeg.add(Conv1D(kernel_size=3, filters=128, activation='relu'))
        branch_eeg.add(Dropout(0.5))
        branch_eeg.add(Conv1D(kernel_size=3, filters=128, activation='relu'))
        branch_eeg.add(Conv1D(kernel_size=3, filters=128, activation='relu'))
        branch_eeg.add(Flatten())
        branch_eeg.add(Dropout(0.5))
        branch_eeg.add(Dense(64, activation='relu'))

        branch_resp = Sequential()
        branch_resp.add(Conv1D(kernel_size=8, filters=64, input_shape=(400, 3), activation='relu'))
        branch_resp.add(Dropout(0.5))
        branch_resp.add(Conv1D(kernel_size=5, filters=128, activation='relu'))
        branch_resp.add(MaxPooling1D(pool_size=2))
        branch_resp.add(Flatten())
        branch_resp.add(Dropout(0.5))
        branch_resp.add(Dense(64, activation='relu'))

        branch_meta = Sequential()
        branch_meta.add(Dense(5, activation='relu', input_dim=5))

        model = Sequential()
        model.add(Merge([branch_eeg, branch_resp, branch_meta], mode='concat'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse', optimizer='rmsprop')
    else:
        if param != 0:
            model = ensemble.GradientBoostingRegressor(n_estimators=param)
        else:
            model = ensemble.GradientBoostingRegressor(n_estimators=100)



    # extract relevant data into appropriate subgroups
    names = df_train.columns

    train_data_x = df_train[names[1:3206]][0:train_number].as_matrix()
    valid_data_x = df_train[names[1:3206]][train_number:total_number].as_matrix()
    test_data_x = df_test[names[1:3206]].as_matrix()

    train_data_eeg = np.reshape(df_train[names[1:2001]][0:train_number].values, (train_number, 2000, 1))
    valid_data_eeg = np.reshape(df_train[names[1:2001]][train_number:total_number].values, (valid_number, 2000, 1))
    test_data_eeg = np.reshape(df_test[names[1:2001]].values, (total_number, 2000, 1))

    train_data_resp = np.reshape(df_train[names[2001:3201]][0:train_number].values, (train_number, 400, 3))
    valid_data_resp = np.reshape(df_train[names[2001:3201]][train_number:total_number].values, (valid_number, 400, 3))
    test_data_resp = np.reshape(df_test[names[2001:3201]].values, (total_number, 400, 3))

    train_data_meta = np.reshape(df_train[names[3201:3206]][0:train_number].values, (train_number, 5))
    valid_data_meta = np.reshape(df_train[names[3201:3206]][train_number:total_number].values, (valid_number, 5))
    test_data_meta = np.reshape(df_test[names[3201:3206]].values, (total_number, 5))

    train_data_y = df_train[names[3206:3207]][0:train_number].values.ravel()
    valid_data_y = df_train[names[3206:3207]][train_number:total_number].values.ravel()

    if convolution:
        print(np.shape(train_data_eeg))
        print(np.shape(train_data_resp))
        print(np.shape(train_data_meta))
        model.fit([train_data_eeg, train_data_resp, train_data_meta], train_data_y,
                  epochs=nb_epoch, validation_data=([valid_data_eeg, valid_data_resp, valid_data_meta], valid_data_y), batch_size=25)
    else:
        model.fit(train_data_x, train_data_y)

    print("Done fitting model!")

    # make predictions
    if convolution:
        train_pred = model.predict([train_data_eeg, train_data_resp, train_data_meta])
        valid_pred = model.predict([valid_data_eeg, valid_data_resp, valid_data_meta])
        test_pred = model.predict([test_data_eeg, test_data_resp, test_data_meta])
    else:
        train_pred = model.predict(train_data_x)
        valid_pred = model.predict(valid_data_x)
        test_pred = model.predict(test_data_x)

    # print results
    train_score = math.sqrt(mean_squared_error(train_data_y, train_pred))
    print('Train Score: %.2f RMSE' % train_score)
    valid_score = math.sqrt(mean_squared_error(valid_data_y, valid_pred))
    print('Validation Score: %.2f RMSE' % valid_score)

    # saves the results to a csv in the "local_info" path
    if saving_results:
        os.chdir(local_info.data_path)
        df_test_pred = pd.DataFrame(test_pred)
        df_test_pred.columns = ['power_increase']
        df_test_pred.index.names = ['index']
        output_name = output_name + ("-VS%.2f.csv" % valid_score)
        df_test_pred.to_csv(output_name, encoding='utf-8')
        print("Stored dataset under the name '%s' at location '%s'" % (output_name, os.getcwd()))
    return train_score, valid_score
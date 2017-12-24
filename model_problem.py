import numpy as np
import pandas as pd
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, Merge, MaxPooling1D
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import local_info
import utils


class dreem_model:
    def __init__(self, settings):
        self.settings = settings
        # load the dataset
        self.df_train, self.df_test \
            = utils.load_dataset(self.settings['small_dataset'], self.settings['storing_small_dataset'])
        # normalize the dataset into [-1, +1]
        df_test = utils.normalize_dataframe(self.df_test)
        df_train = utils.normalize_dataframe(self.df_train)

        # choose validation and training data from the training set
        total_number = len(df_train)
        valid_number = round(self.settings['validation_share'] * total_number)
        train_number = total_number - valid_number

        # extract relevant data into appropriate subgroups
        names = df_train.columns

        self.train_data_x = df_train[names[1:3206]][0:train_number].as_matrix()
        self.valid_data_x = df_train[names[1:3206]][train_number:total_number].as_matrix()
        self.test_data_x = df_test[names[1:3206]].as_matrix()

        self.train_data_eeg \
            = np.reshape(df_train[names[1:2001]][0:train_number].values, (train_number, 2000, 1))
        self.valid_data_eeg \
            = np.reshape(df_train[names[1:2001]][train_number:total_number].values, (valid_number, 2000, 1))
        self.test_data_eeg \
            = np.reshape(df_test[names[1:2001]].values, (total_number, 2000, 1))

        self.train_data_resp \
            = np.reshape(df_train[names[2001:3201]][0:train_number].values, (train_number, 400, 3))
        self.valid_data_resp \
            = np.reshape(df_train[names[2001:3201]][train_number:total_number].values, (valid_number, 400, 3))
        self.test_data_resp \
            = np.reshape(df_test[names[2001:3201]].values, (total_number, 400, 3))

        self.train_data_meta \
            = np.reshape(df_train[names[3201:3206]][0:train_number].values, (train_number, 5))
        self.valid_data_meta \
            = np.reshape(df_train[names[3201:3206]][train_number:total_number].values, (valid_number, 5))
        self.test_data_meta \
            = np.reshape(df_test[names[3201:3206]].values, (total_number, 5))

        self.train_data_y = df_train[names[3206:3207]][0:train_number].values.ravel()
        self.valid_data_y = df_train[names[3206:3207]][train_number:total_number].values.ravel()

        self.results = []

    def apply_model(self, param):
        # choose between possible models
        if self.settings['model_option'] == 'convolution':
            train_pred, valid_pred, test_pred, train_score, valid_score = self.convolution_model()
        elif self.settings['model_option'] == 'gradient_boost':

            for p in param:
                train_pred, valid_pred, test_pred, train_score, valid_score = self.gradient_boost_model(p)
                self.results += [[p, train_score, valid_score]]

        # saves the results to a csv in the "local_info" path
        if 'test_pred' not in locals():
            print("Problem!")
        else:
            if self.settings['saving_results']:
                os.chdir(local_info.data_path)
                df_test_pred = pd.DataFrame(test_pred)
                df_test_pred.columns = ['power_increase']
                df_test_pred.index.names = ['index']
                output_name = self.settings['output_name'] + ("-VS%.2f.csv" % valid_score)
                df_test_pred.to_csv(output_name, encoding='utf-8')
                print("Stored dataset under the name '%s' at location '%s'" % (output_name, os.getcwd()))
            return train_score, valid_score

    def convolution_model(self):
        # define model
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

        # fit model
        model.fit([self.train_data_eeg, self.train_data_resp, self.train_data_meta],
                  self.train_data_y, epochs=self.settings['nb_epoch'],
                  validation_data=(
                      [self.valid_data_eeg, self.valid_data_resp, self.valid_data_meta], self.valid_data_y
                  ),
                  batch_size=25)

        # test prediction
        train_pred = model.predict([self.train_data_eeg, self.train_data_resp, self.train_data_meta])
        valid_pred = model.predict([self.valid_data_eeg, self.valid_data_resp, self.valid_data_meta])
        test_pred = model.predict([self.test_data_eeg, self.test_data_resp, self.test_data_meta])

        # print results
        train_score = math.sqrt(mean_squared_error(self.train_data_y, train_pred))
        valid_score = math.sqrt(mean_squared_error(self.valid_data_y, valid_pred))
        print('Train Score: %.2f RMSE \n Validation Score: %.2f RMSE' % (train_score, valid_score))

        return train_pred, valid_pred, test_pred, train_score, valid_score

    def gradient_boost_model(self, parameter):
        model = ensemble.GradientBoostingRegressor( n_estimators=parameter)
        model.fit(self.train_data_x, self.train_data_y)
        train_pred = model.predict(self.train_data_x)
        if len(self.valid_data_y != 0):
            valid_pred = model.predict(self.valid_data_x)
        else:
            valid_pred = 0
        test_pred = model.predict(self.test_data_x)

        # print results
        train_score = math.sqrt(mean_squared_error(self.train_data_y, train_pred))
        if len(self.valid_data_y != 0):
            valid_score = math.sqrt(mean_squared_error(self.valid_data_y, valid_pred))
        else:
            valid_score = 0
        print('Param: %s \nTrain Score: %.2f RMSE \nValidation Score: %.2f RMSE\n' % (parameter, train_score, valid_score))

        return train_pred, valid_pred, test_pred, train_score, valid_score

    def describe_self(self):
        print('\n Results for parameter values (MSE)')
        for res in self.results:
            print('-> %s: training %.3f - validaton %.3f' % (res[0], res[1], res[2]))

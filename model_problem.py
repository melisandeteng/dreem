import numpy as np
import pandas as pd
import lightgbm as lgb
import math
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, Merge, MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import local_info
import utils
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import matplotlib.pyplot as plt
from scipy.fftpack import fftn


class dreem_model:
    def __init__(self, settings):
        self.settings = settings

        # load the dataset
        self.df_train, self.df_test \
            = utils.load_dataset(self.settings['small_dataset'], self.settings['storing_small_dataset'])

        names = self.df_train.columns
        # IS THIS USEFUL???
        # normalize the eeg / resp dataset into [-1, +1]
        quantile = False
        if quantile:
            q_transform_eeg = QuantileTransformer()
            q_transform_resp = QuantileTransformer()
            self.df_train[names[1:2001]] = q_transform_eeg.fit_transform(self.df_train[names[1:2001]])
            self.df_train[names[2001:3201]] = q_transform_resp.fit_transform(self.df_train[names[2001:3201]])
            self.df_test[names[1:2001]] = q_transform_eeg.transform(self.df_test[names[1:2001]])
            self.df_test[names[2001:3201]] = q_transform_resp.transform(self.df_test[names[2001:3201]])
        else:
            df_c_eeg = pd.concat((self.df_train[names[1:2001]], self.df_test[names[1:2001]]))
            df_c_resp = pd.concat((self.df_train[names[2001:3201]], self.df_test[names[2001:3201]]))

            m_eeg = df_c_eeg.mean().mean()
            m_resp = df_c_resp.mean().mean()
            v_eeg = df_c_eeg.var().mean()
            v_resp = df_c_resp.var().mean()

            self.df_train[names[1:2001]] = utils.normalize_dataframe(self.df_train[names[1:2001]], m_eeg, v_eeg)
            self.df_train[names[2001:3201]] = utils.normalize_dataframe(self.df_train[names[2001:3201]], m_resp, v_resp)
            self.df_test[names[1:2001]] = utils.normalize_dataframe(self.df_test[names[1:2001]], m_eeg, v_eeg)
            self.df_test[names[2001:3201]] = utils.normalize_dataframe(self.df_test[names[2001:3201]], m_resp, v_resp)
        # choose validation and training data from the training set
        total_number = len(self.df_train)
        valid_number = round(self.settings['validation_share'] * total_number)
        train_number = total_number - valid_number

        # extract relevant data into appropriate subgroups
        self.train_data_x = self.df_train[names[1:3206]][0:train_number].as_matrix()
        self.valid_data_x = self.df_train[names[1:3206]][train_number:total_number].as_matrix()
        self.test_data_x = self.df_test[names[1:3206]].as_matrix()

        self.train_data_eeg \
            = np.reshape(self.df_train[names[1:2001]][0:train_number].values, (train_number, 2000, 1))
        self.valid_data_eeg \
            = np.reshape(self.df_train[names[1:2001]][train_number:total_number].values, (valid_number, 2000, 1))
        self.test_data_eeg \
            = np.reshape(self.df_test[names[1:2001]].values, (total_number, 2000, 1))

        self.train_data_resp \
            = np.reshape(self.df_train[names[2001:3201]][0:train_number].values, (train_number, 400, 3))
        self.valid_data_resp \
            = np.reshape(self.df_train[names[2001:3201]][train_number:total_number].values, (valid_number, 400, 3))
        self.test_data_resp \
            = np.reshape(self.df_test[names[2001:3201]].values, (total_number, 400, 3))

        # One Hot Encoding for users
        users_train = np.reshape(self.df_train[names[3204:3205]].values, (total_number, 1))
        users_test = np.reshape(self.df_test[names[3204:3205]].values, (total_number, 1))
        enc = OneHotEncoder()
        enc.fit(users_train)
        encoded_users_train = enc.transform(users_train).toarray()
        encoded_users_test = enc.transform(users_test).toarray()
        #print("Number of different users (train): %s" % np.shape(encoded_users_train)[1])
        #print("Number of different users (test): %s" % np.shape(encoded_users_test)[1])

        self.train_data_meta = np.concatenate(
            (np.reshape(self.df_train[names[3201:3204]][0:train_number].values, (train_number, 3)),
             np.reshape(self.df_train[names[3205:3206]][0:train_number].values, (train_number, 1)),
             encoded_users_train[0:train_number]), axis=1)
        self.valid_data_meta = np.concatenate(
            (np.reshape(self.df_train[names[3201:3204]][train_number:total_number].values, (valid_number, 3)),
             np.reshape(self.df_train[names[3205:3206]][train_number:total_number].values, (valid_number, 1)),
             encoded_users_train[train_number:total_number]), axis=1)
        self.test_data_meta = np.concatenate(
            (np.reshape(self.df_test[names[3201:3204]].values, (total_number, 3)),
             np.reshape(self.df_test[names[3205:3206]].values, (total_number, 1)),
             encoded_users_test), axis=1)

        self.train_data_y = self.df_train[names[3206:3207]][0:train_number].values.ravel()
        self.valid_data_y = self.df_train[names[3206:3207]][train_number:total_number].values.ravel()

        self.results = []

        if settings['test_fft']:
            y = np.fft.rfft(self.train_data_resp, 20)
            print("shape after rfft: %s" % str(np.shape(y)))
            X = np.fft.fftn(self.train_data_resp)
            print("shape after fftn: %s" % str(np.shape(X)))
            print("shape before fft: %s" % str(np.shape(self.train_data_resp)))
            print(np.shape(self.train_data_resp[:, :, 0]))
            z = np.apply_along_axis(lambda x: np.fft.rfft(x)[0:20], 1, self.train_data_resp[:, :, 0])
            print("shape after RS: %s" % str(np.shape(z)))

    def apply_model(self):
        # choose between possible models
        if self.settings['model_option'] == 'convolution':
            print("\nModel chosen: CNN")
            branch_eeg,\
            train_pred, valid_pred, test_pred, train_score, valid_score = self.convolution_model()
            str_res = 'CNN'

        elif self.settings['model_option'] == 'gradient_boost':
            print("\nModel chosen: gradient boosting")
            train_pred, valid_pred, test_pred, train_score, valid_score = self.simple_gradient_boost_model(
                self.train_data_x, self.train_data_y, self.valid_data_x, self.valid_data_y, self.test_data_x)
            str_res = str(self.settings['grad_boost_param']) + '-gradient boost'

        elif self.settings['model_option'] == 'convolution + gradient_boost':
            print("\nModel chosen: CNN + gradient boost on learned features")
            train_pred, valid_pred, test_pred, train_score, valid_score = self.conv_grad_model()
            str_res = 'Conv + '+str(self.settings['grad_boost_param'])+'-gradient boost'
        self.results += [[str_res, train_score, valid_score]]

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
        branch_eeg.add(Conv1D(kernel_size=2000, filters=64, input_shape=(2000, 1), activation='relu',
                              kernel_regularizer=l2(self.settings['regularization_param'])))
        branch_eeg.add(Dropout(0.5))
        branch_eeg.add(Dense(64, activation='relu', kernel_regularizer=l2(self.settings['regularization_param'])))
        branch_eeg.add(Dropout(0.5))
        branch_eeg.add(Dense(64, activation='relu', kernel_regularizer=l2(self.settings['regularization_param'])))
        branch_eeg.add(Flatten())
        branch_eeg.add(Dense(1, activation='relu', kernel_regularizer=l2(self.settings['regularization_param'])))

        optimizer = 'rmsprop'
        # optimizer = Adam(lr=self.settings['adam_lr'])
        branch_eeg.compile(loss='mse', optimizer=optimizer)

        # fit model
        history = branch_eeg.fit(self.train_data_eeg, self.train_data_y,
                            epochs=self.settings['nb_epoch'], batch_size=250,
                            validation_data=(self.valid_data_eeg,
                                             self.valid_data_y))

        if self.settings['display']:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

        print(branch_eeg.summary())

        # test prediction
        train_pred = branch_eeg.predict(self.train_data_eeg)
        valid_pred = branch_eeg.predict(self.valid_data_eeg)
        test_pred = branch_eeg.predict(self.test_data_eeg)

        # print results
        train_score = math.sqrt(mean_squared_error(self.train_data_y, train_pred))
        valid_score = math.sqrt(mean_squared_error(self.valid_data_y, valid_pred))
        print('CNN! Train Score: %.2f RMSE \n Validation Score: %.2f RMSE' % (train_score, valid_score))

        return branch_eeg, train_pred, valid_pred, test_pred, train_score, valid_score

    def light_gradient_boost_model(self, train_data_x, train_data_y, valid_data_x, valid_data_y, test_data_x):
        model = lgb.LGBMRegressor(objective = 'regressor', num_leaves = 31, learning_rate = 0.05, n_estimators = 100)
        model.fit(train_data_x, train_data_y, eval_set=[(valid_data_x, valid_data_y)],
        eval_metric='l1', early_stopping_rounds = 5)
        print('Start prediction,...')
        train_pred = model.predict(train_data_x, num_iteration = model.best_iteration_ )

        # print results
        if len(valid_data_y != 0):
            valid_pred = model.predict(valid_data_x)
        else:
            valid_pred = 0
        test_pred = model.predict(test_data_x)
        train_score = math.sqrt(mean_squared_error(train_data_y, train_pred))
        if len(valid_data_y != 0):
            valid_score = math.sqrt(mean_squared_error(valid_data_y, valid_pred))
        else:
            valid_score = 0
        print('Gradient boost! Param: %s \nTrain Score: %.2f RMSE \nValidation Score: %.2f RMSE\n' % (self.settings['grad_boost_param'], train_score, valid_score))

        return train_pred, valid_pred, test_pred, train_score, valid_score


    def simple_gradient_boost_model(self, train_data_x, train_data_y, valid_data_x, valid_data_y, test_data_x):
        # model = ensemble.AdaBoostRegressor(n_estimators=500)
        model = ensemble.GradientBoostingRegressor(n_estimators=self.settings['grad_boost_param'],
                                                   learning_rate=self.settings['grad_boost_lr'],
                                                   subsample=self.settings['grad_boost_subsample'],
                                                   max_features=self.settings['grad_boost_max_features'])

        model.fit(train_data_x, train_data_y)
        train_pred = model.predict(train_data_x)
        if len(valid_data_y != 0):
            valid_pred = model.predict(valid_data_x)
        else:
            valid_pred = 0
        test_pred = model.predict(test_data_x)

        # print results
        train_score = math.sqrt(mean_squared_error(train_data_y, train_pred))
        if len(valid_data_y != 0):
            valid_score = math.sqrt(mean_squared_error(valid_data_y, valid_pred))
        else:
            valid_score = 0
        print('Gradient boost! Param: %s \nTrain Score: %.2f RMSE \nValidation Score: %.2f RMSE\n' % (self.settings['grad_boost_param'], train_score, valid_score))

        return train_pred, valid_pred, test_pred, train_score, valid_score

    def conv_grad_model(self):
        if self.settings['use_stored_activations'] and self.settings['should_store_activations'] is False:
            activations_train, activations_valid, activations_test = utils.load_activations()
        else:
            branch_eeg, train_pred, valid_pred, test_prediction, train_score, valid_score = self.convolution_model()

            optimizer = 'rmsprop'
            # optimizer = Adam(lr=self.settings['adam_lr'])
            branch_eeg.compile(loss='mse', optimizer=optimizer)

            truncated_branch_eeg = Sequential()
            truncated_branch_eeg.add(Conv1D(kernel_size=2000, filters=64, input_shape=(2000, 1), activation='relu',
                                       weights=branch_eeg.layers[0].get_weights()))
            truncated_branch_eeg.add(Dropout(0.5))
            truncated_branch_eeg.add(Dense(64, activation='relu',
                                       weights=branch_eeg.layers[2].get_weights()))
            truncated_branch_eeg.add(Dropout(0.5))
            truncated_branch_eeg.add(Dense(64, activation='relu',
                                       weights=branch_eeg.layers[4].get_weights()))
            truncated_branch_eeg.add(Flatten())

            activations_train = truncated_branch_eeg.predict(self.train_data_eeg)
            activations_valid = truncated_branch_eeg.predict(self.valid_data_eeg)
            activations_test = truncated_branch_eeg.predict(self.test_data_eeg)

            if self.settings['should_store_activations']:
                if self.settings['small_dataset']:
                    print("\nI'm not going to store the small dataset's activations, that makes no sense!\n")
                else:
                    utils.store_activations(activations_train, activations_valid, activations_test)

        if self.settings['do_not_use_raw_features']:
            if self.settings['small_dataset']:
                train_concat = activations_train[0:round((1 - self.settings['validation_share']) * len(self.df_train))]
                valid_concat = activations_valid[0:round(self.settings['validation_share'] * len(self.df_train))]
                test_concat = activations_test[0:len(self.df_train)]
            else:
                train_concat = np.concatenate(
                    (activations_train, np.reshape(self.train_data_meta, (-1, np.shape(self.train_data_meta)[1])),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.train_data_resp[:, :, 0]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.train_data_resp[:, :, 1]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.train_data_resp[:, :, 2])), axis=1)
                # print(np.shape(self.valid_data_meta))
                # print(np.shape(np.reshape(self.valid_data_meta, (-1, np.shape(self.valid_data_meta)[1]))))
                # print(np.shape(np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                #                          self.valid_data_resp[:, :, 0])))
                # print(np.shape(np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                #                          self.valid_data_resp[:, :, 1])))
                # print(np.shape(np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                #                          self.valid_data_resp[:, :, 2])))
                valid_concat = np.concatenate(
                    (activations_valid, np.reshape(self.valid_data_meta, (-1, np.shape(self.valid_data_meta)[1])),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.valid_data_resp[:, :, 0]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.valid_data_resp[:, :, 1]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.valid_data_resp[:, :, 2])), axis=1)
                test_concat = np.concatenate(
                    (activations_test, np.reshape(self.test_data_meta, (-1, np.shape(self.train_data_meta)[1])),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.test_data_resp[:, :, 0]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.test_data_resp[:, :, 1]),
                     np.apply_along_axis(lambda x: np.fft.rfft(x)[0:self.settings['fft_setting']], 1,
                                         self.test_data_resp[:, :, 2])), axis=1)
                print("Using activations + OHE + FFT (total train dimension: %s" % str(np.shape(train_concat)))
        else:
            if self.settings['small_dataset']:
                train_concat = activations_train[0:round((1-self.settings['validation_share']) * len(self.df_train))]
                valid_concat = activations_valid[0:round(self.settings['validation_share'] * len(self.df_train))]
                test_concat = activations_test[0:len(self.df_train)]
            else:
                train_concat = np.concatenate(
                    (activations_train, np.reshape(self.train_data_x, (-1, np.shape(self.train_data_x)[1]))), axis=1)
                valid_concat = np.concatenate(
                    (activations_valid, np.reshape(self.valid_data_x, (-1, np.shape(self.train_data_x)[1]))), axis=1)
                test_concat = np.concatenate(
                    (activations_test, np.reshape(self.test_data_x, (-1, np.shape(self.train_data_x)[1]))), axis=1)
                print("Using activations + raw features")

        # apply gradient boost model
        train_pred, valid_pred, test_pred, train_score, valid_score = self.light_gradient_boost_model(
            train_concat, self.train_data_y, valid_concat, self.valid_data_y, test_concat)

        # print results
        train_score = math.sqrt(mean_squared_error(self.train_data_y, train_pred))
        valid_score = math.sqrt(mean_squared_error(self.valid_data_y, valid_pred))

        return train_pred, valid_pred, test_pred, train_score, valid_score

    def describe_self(self):
        print('\n Results for parameter values (MSE)')
        for res in self.results:
            print('-> %s: training %.3f - validation %.3f' % (res[0], res[1], res[2]))

import pandas as pd
import numpy as np
import os
import local_info


def load_activations():
    print("Reading activations dataset...")
    activations_train = np.loadtxt(local_info.data_path + 'activations_train.csv', delimiter=",")
    activations_valid = np.loadtxt(local_info.data_path + 'activations_valid.csv', delimiter=",")
    activations_test = np.loadtxt(local_info.data_path + 'activations_test.csv', delimiter=",")
    print("Activations have been read from csv! (train shape is %s)" % str(np.shape(activations_train)))
    return activations_train, activations_valid, activations_test


def store_activations(activ_train, activ_valid, activ_test):
    os.chdir(local_info.data_path)
    np.savetxt('activations_train.csv', activ_train, delimiter=",")
    np.savetxt('activations_valid.csv', activ_valid, delimiter=",")
    np.savetxt('activations_test.csv', activ_test, delimiter=",")
    print("Activations have been stored! (train shape is %s)\n" % str(np.shape(activ_train)))


def load_dataset(small_dataset, storing_small_dataset):
    print("Reading raw dataset...")
    if small_dataset:
        print("Small dataset chosen")
        df_train = pd.read_csv(local_info.data_path + 'extract_train.csv')
        df_test = pd.read_csv(local_info.data_path + 'extract_test.csv')
    else:
        print("Full dataset chosen")
        df_train = pd.read_csv(local_info.data_path + 'train.csv')
        df_test = pd.read_csv(local_info.data_path + 'test.csv')
    print("Done reading dataset! \n")

    # stores an extract of 500 rows instead of 50k for debugging purposes
    if storing_small_dataset:
        store_small_dataset(df_train, 500, "train")
        store_small_dataset(df_test, 500, "test")

    return df_train, df_test


# normalizes all columns except last one (power increase)
def normalize_dataframe(df, mean, var):
    df_out = df.sub(mean, axis=0)
    df_out = df_out.div(var, axis=0)
    return df_out


def store_small_dataset(df, length, name):
    print("Storing extract of " + name + " data...")
    os.chdir(local_info.data_path)
    temp_df = df[0:length]
    temp_df.to_csv('extract_' + name + '.csv', encoding='utf-8', index=False)
    print("Done storing %s extract at location %s ! \n" % (name, os.getcwd()))


def merge_solutions(path, array):
    os.chdir(path)
    df_array = []
    name = "concat"
    for file_name in array:
        df = pd.read_csv(file_name)
        df_array += [df['power_increase']]
        name += "-" + file_name
    df_concat = pd.concat(df_array)
    df_by_row = df_concat.groupby(df_concat.index)
    df_means = df_by_row.mean()
    df_means.columns = ['power_increase']
    df_means.index.names = ['index']
    df_means.to_csv(name, encoding='utf-8')
    print("Stored dataset under the name '%s' at location '%s'" % (name, os.getcwd()))
    return df_means


# merge_solutions('D:/Joel/1.2017-3A/1.OMA/MLC-dreem',['convnet _13-12-2017-1h43-VS0.67.csv','gboost_11-12-2017-18h12.csv'])

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import local_info


def load_dataset(small_dataset, storing_small_dataset):
    print("Reading dataset...")
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
def normalize_dataframe(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    names = df.columns
    if len(names) <= 3206:
        df_out = pd.DataFrame(scaler.fit_transform(df.values), columns=names)
    else:
        df_out = pd.DataFrame(scaler.fit_transform(df[df.columns[0:3206]].values), columns=names[0:3206])
        df_out['power_increase'] = df['power_increase']
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

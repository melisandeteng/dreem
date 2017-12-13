import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import local_info


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

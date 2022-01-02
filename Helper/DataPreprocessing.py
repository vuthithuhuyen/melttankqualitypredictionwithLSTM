import numpy
import pandas as pd
from numpy import argmax
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

from Model import GlobalVariables
from Model.GlobalVariables import raw_data_file, scaler


# Read the raw data into data frame df
# Transform features in df to X and output to y
from Model.DeeplearningModel import SaveDictToFile


def ReadData():
    df = pd.read_csv(raw_data_file)
    df.dropna(inplace=True)
    df.drop('NUM', axis='columns', inplace=True)
    # df = df[(df.MIXA_PASTEUR_STATE == 0) | (df.MIXA_PASTEUR_STATE == 1)]
    #
    # # Temprature 800 => 80.0
    # df.iloc[:, 3] /= 10
    # df.iloc[:, 4] /= 10
    #
    # X = df[inputfeatures]
    # y = df[output]
    # print(len(X))
    return df, None, None


# Scale data
def ScaleData(X, y):
    X = X.astype('float32')
    X_transformed = scaler.fit_transform(X)

    # Create dictionaries for labels in y
    output_set = set(y)
    int_to_name = {k: w for k, w in enumerate(output_set)}
    name_to_int = {w: k for k, w in int_to_name.items()}

    # encode values of y
    y_encoded = []
    for i in range(0, len(y)):
        y_encoded.append(one_hot_encode(name_to_int, y[i]))

    y_encoded = numpy.asarray(y_encoded)
    return X_transformed, y_encoded, name_to_int, int_to_name


# one hot encode sequence
def one_hot_encode(name_to_int_dict, value):
    vector = [0 for _ in range(len(name_to_int_dict))]
    key = name_to_int_dict[value]
    vector[key] = 1
    return vector

# vector to label:
def vector_to_label(vector, _dict):
    for i in range(len(vector)):
        if vector[i]==1:
            return _dict[i]
    return None


def one_hot_decode(encoded_seq):
    result = [argmax(vector) for vector in encoded_seq]
    return result


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        insert_df = data.shift(i)
        insert_df = insert_df[insert_df.columns[:-n_out]]
        cols.append(insert_df)
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars-1)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        insert_df = data.shift(i)
        insert_df = insert_df[insert_df.columns[-n_out]]
        cols.append(insert_df)
        if i == 0:
            names += [('output%d(t)' % (j + 1)) for j in range(n_out)]
        else:
            names += [('output%d(t+%d)' % (j + 1, i)) for j in range(n_out)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    values = agg.values
    return values[:, :-1], values[:, -1]
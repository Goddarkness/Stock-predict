import pandas as pd
import numpy as np
from numpy import array, hstack

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pickle

previous_days = 7


def dataprocessing(csv, split_date, mode="train", ticker=None):
    dataset = pd.read_csv(csv, index_col="date", parse_dates=True)
    dataset['close'].shift(previous_days)
    dataset = dataset[(dataset['ticker'] == '^GSPC') | (
        dataset['ticker'] == '^DJI') | (dataset['ticker'] == '^GDAXI')]
    df_dummies = pd.get_dummies(dataset['ticker'], prefix='ticker')
    dataset = pd.concat([dataset, df_dummies], axis=1)

    dataset = dataset.drop(columns=['ticker'])
    train = dataset.loc[dataset.index < split_date]
    test = dataset.loc[dataset.index >= split_date]
    test = test[test[f'ticker_{ticker}'] == 1]

    if mode == 'train':
        dataset = train
    else:
        dataset = test

    x_cols = ['open', 'low', 'volume', 'ticker_^DJI',
              'ticker_^GSPC', 'ticker_^GDAXI']
    x = dataset[x_cols].values
    feature = np.column_stack(x)
    feature = feature.reshape(dataset.shape[0], -1)
    y = dataset['close'].values
    y = y.reshape(dataset.shape[0], 1)
    print(feature[0])

    if mode == 'train':
        scaler = MinMaxScaler(feature_range=(-1, 1))

        feature_scaled = scaler.fit_transform(feature)

        pickle.dump(scaler, open('xscaler.pkl', 'wb'))

        yscaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = yscaler.fit_transform(y)
        pickle.dump(yscaler, open('yscaler.pkl', 'wb'))

        feature_scaled = np.column_stack((feature_scaled, y_scaled))
        return feature_scaled

    else:
        scaler = pickle.load(open("xscaler.pkl", 'rb'))
        feature_scaled = scaler.fit_transform(feature)
        yscaler = pickle.load(open("yscaler.pkl", 'rb'))
        y_scaled = scaler.fit_transform(y)
        # return feature_scaled[:, :-1] ,feature_scaled[:, -1]
        return feature_scaled, y_scaled


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-
                                 1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


# spliting the dataset--------------------------
# Splitting the dataset into the Training set and Test set
def split_dataset(X, y):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0)
    return train_X, train_y, test_X, test_y

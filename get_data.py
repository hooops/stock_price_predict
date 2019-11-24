import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_df(df_path, columns):
    df_list = []
    files = sorted(os.listdir(df_path))
    for f in files:
        df_year = pd.read_csv(os.path.join(df_path, f), encoding="shift-jis", header=1)
        df_list.append(df_year)
    for i, df_year in enumerate(df_list):
        if i == 0:
            df = df_year.copy()
        else:
            df = pd.concat([df, df_year])

    df.columns = columns

    return df


def add_index(series, ix):
    series.index = pd.to_datetime(ix)
    return series


def get_data(df_path, columns, valid_len):
    df = get_df(df_path, columns)
    train, valid = df[:-valid_len], df[-valid_len:]
    data = df['Closing_price'].values.reshape(-1, 1)
    train = train['Closing_price'].values.reshape(-1, 1)
    valid = valid['Closing_price'].values.reshape(-1, 1)
    data = np.array(data, dtype='float')
    train = np.array(train, dtype='float')
    valid = np.array(valid, dtype='float')
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    data = scaler.transform(data)
    train = torch.Tensor(train).view(-1)
    valid = torch.Tensor(valid).view(-1)
    data = torch.Tensor(data).view(-1)

    return data, train, valid, scaler


def generate_dataset(input_data, seq_length):
    output_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i : i+seq_length]
        train_label = input_data[i+seq_length : i+seq_length+1]
        output_seq.append((train_seq, train_label))

    return output_seq
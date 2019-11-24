import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import mean_squared_error
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from get_data import get_df, add_index, get_data
from model import LSTM
from config import config
from utils import seed_everything


config = config()
DATA_PATH = config.path.data_path
columns = config.csv_columns
valid_len = config.length.valid_len
seq_length = config.length.seq_length
fut_pred = config.length.fut_pred
device = config.device
seed = config.seed
model_path = config.path.model_path
fig_path = config.path.fig_path
pred_path = config.path.pred_path

seed_everything(seed)

df = get_df(DATA_PATH, columns).reset_index(drop=True)
price = df['Closing_price']
df_len = len(df)

data, _, _, scaler = get_data(DATA_PATH, columns, valid_len)

test_inputs = data[-seq_length:].tolist()

model = LSTM()
model.to(device)
model.load_state_dict(torch.load(model_path))


def main():
    for _ in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-seq_length:])
        seq = seq.to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))
            test_inputs.append(model(seq).item())

    pred = scaler.inverse_transform(np.array(test_inputs[-fut_pred:]).reshape(-1, 1))
    with open(pred_path, 'wb') as f:
        pickle.dump(pred, f)

    x = np.arange(df_len, df_len+fut_pred, 1)
    plt.figure(figsize=(14, 4))
    plt.title('Stock Price Predict', fontsize=16)
    plt.plot(price, label='Closing Price')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(x, pred.flatten(), label='pred')
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
    plt.savefig(os.path.join(fig_path, 'prediction_plot.png'))


if __name__ == "__main__":
    main()
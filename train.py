import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from get_data import get_df, add_index, get_data, generate_dataset
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
lr = config.parameter.lr
epochs = config.parameter.epochs
model_path = config.path.model_path
fig_path = config.path.fig_path

seed_everything(seed)

df = get_df(DATA_PATH, columns).reset_index(drop=True)
price = df['Closing_price']
train_len = len(df) - valid_len

_, train, valid, scaler = get_data(DATA_PATH, columns, valid_len)
train_loader = generate_dataset(train, seq_length)
valid_loader = generate_dataset(valid, seq_length)

test_inputs = train[-seq_length:].tolist()

model = LSTM()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=5)

def main():
    train_losses = []
    valid_losses = []
    best_loss = np.inf
    for i in range(epochs):
        train_loss = 0
        valid_loss = 0
        if i % 10 == 0:
            print('-----------------------')
            print(f'epoch: {i+1} / {epochs}')
            print('-----------------------')
        for seq, label in (train_loader):
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            pred = model(seq)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        if i % 10 == 0:
            print('Train Loss: {:.6f}'.format(train_loss))

        for seq, label in valid_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            pred = model(seq)
            loss = criterion(pred, label)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        if i % 10 == 0:
            print('Validation Loss: {:.6f}'.format(valid_loss), '\n\n')

        if valid_loss < best_loss:
            print(f'epoch: {i+1}')
            print('Validation MSE {:.6f} --> {:.6f}     Saving Model......\n\n'.format(best_loss, valid_loss))
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        scheduler.step(valid_loss)

    plt.figure(figsize=(10, 4))
    plt.title('Loss Plot', fontsize=16)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(fig_path, 'loss_plot.png'))

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for _ in range(fut_pred):
        seq = torch.Tensor(test_inputs[-seq_length:])
        seq = seq.to(device)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))
            test_inputs.append(model(seq).item())
    pred = scaler.inverse_transform(np.array(test_inputs[-fut_pred:]).reshape(-1, 1))

    x = np.arange(train_len, train_len+fut_pred, 1)
    plt.figure(figsize=(14, 4))
    plt.title('Stock Price Predict', fontsize=16)
    plt.plot(price, label='Closing Price')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(x, pred.flatten(), label='pred')
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
    plt.savefig(os.path.join(fig_path, 'validation_plot.png'))


if __name__ == "__main__":
    main()
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from get_data import get_df, add_index
from config import config


config = config()
DATA_PATH = config.path.data_path
columns = config.csv_columns
fig_path = config.path.fig_path

def main():
    df = get_df(DATA_PATH, columns)
    opening = add_index(df['Opening_price'], df['Datetime'])
    high = add_index(df['High_price'], df['Datetime'])
    low = add_index(df['Low_price'], df['Datetime'])
    closing = add_index(df['Closing_price'], df['Datetime'])

    price_dic = {
        0: 'Opening',
        1: 'High',
        2: 'Low',
        3: 'Closing',
    }

    plt.figure(figsize=(12, 4))
    plt.title('Stock Price', fontsize=16)
    for i, price in enumerate([opening, high, low, closing]):
        plt.plot(price, label=f'{price_dic[i]} Price', linestyle='-.')
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
    plt.savefig(os.path.join(fig_path, 'stock_price.png'));

    plt.figure(figsize=(12, 4))
    plt.title('Previous day difference (Closing Price)', fontsize=16)
    plt.plot(closing.diff())
    plt.savefig(os.path.join(fig_path, 'previous_day_difference.png'))


if __name__ == "__main__":
    main()
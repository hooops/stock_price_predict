Name
====
stock_price_prediction

### Overview
- Forecasting with stock price using DeepLearning (LSTM)

### Dataset
- Download the CSV file from the following site.
- https://kabuoji3.com/stock/
- The above site distributes stock price data of Japanese companies.
- The first line is not read when converting the original CSV file to pandas dataframe.
- Code changes are required when using datasets with different formats.

### Usage
```
python plot.py  # Outputs a plot of the change in stock price and the previous day's difference between the closing prices.
python train.py  # train
python eval.py  # eval
```

### Install
```
pip install torch
pip install python-box
```

### Others
#### config.yml
- Each variable is defined. Please You rewrite as necessary.

#### data directory
- It is possible to store data for multiple years.

#### using data
- The first half of the given data is used for training and the second half (valid_len) is used for validation.
- In general, after validating the model, learning the validation data be able to predict the future more accurately.
- This code does not implement the above technique.
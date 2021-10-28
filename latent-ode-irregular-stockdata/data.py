# 1. Get Open and Close Price of asset (o, c) for each trading day.
# libraries
from pandas_datareader import data as pdr
import yfinance as yf
import os
import pandas as pd
import numpy as np
from pandas import concat

def download_raw_stock_data(filepath, tickers, start, end, period = '1d'):
    """
    Download Stock tickers
    :Parameters:
        filepath: str
            path to store the raw data
        tickers : str, list
            List of tickers to download
        period: str
            the frequency at which to gather the data; common options would include ‘1d’ (daily), ‘1mo’ (monthly), ‘1y’ (yearly)
        start: str
            the date to start gathering the data. For example ‘2010–1–1’
        end: str
            the date to end gathering the data. For example ‘2020–1–25’
    
    """
    #define the ticker symbol
    tickerSymbol = tickers

    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    #get the historical prices for this ticker
    tickerDf = tickerData.history(period=period, start=start, end=end)
    tickerDf.to_csv(filepath)

def stockDataTransformer(filepath):
    df = pd.read_csv(filepath)
    df.set_index('Date', inplace=True)
    df1 = df[['Open', 'Close']].copy()
    data = df1.values
    n_samples = data.shape[0]//10*10
    reshape_number = n_samples*data.shape[1]//10
    data1 = data[:n_samples].reshape((reshape_number, 10))
    return data1

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def split_data(perc_train, perc_valid, lag, data_orig, data_m1, n_features_orig, n_features_median):
    values = data_m1
    
    sizeOfReframed = len(data_m1)
    len_train = int(perc_train*sizeOfReframed) # int(sizeOfReframed - len_test) # - len_valid)
    train_data_orig = data_orig[:len_train, :]
    # valid = values[len_train:len_valid+len_train, :]
    test_data_orig = data_orig[len_train:, :]  # [len_valid+len_train:, :]
    # n_features = n_features
    
    train_data_ml = values[:len_train, :]
    test_data_ml = values[len_train:, :] 
    # split into input and outputs
    n_obs = lag * n_features_orig
    n_obs_median = (lag+forecast) * n_features_median
    train_X, train_y = train_data_orig[:, :n_obs], train_data_ml[:, :n_obs_median]
    test_X, test_y = test_data_orig[:, :n_obs], test_data_ml[:, :n_obs_median]
    # valid_X, valid_y = valid[:, :n_obs], valid[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)
    
    # reshape input to be 3D [samples, features, lag]
    train_X = train_X.reshape((train_X.shape[0], n_features_orig, lag))
    test_X = test_X.reshape((test_X.shape[0], n_features_orig, lag))
    # valid_X = valid_X.reshape((valid_X.shape[0], lag, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)  # , valid_X.shape, valid_y.shape)
    
    # Get the reconstruction train_y, test_y and extrapolated train_y, test_y
    train_y_recon, train_y_extrapol = train_y[:, :lag], train_y[:, lag:]
    test_y_recon, test_y_extrapol = test_y[:, :lag], test_y[:, lag:]
    dataload = {
        'train_data_orig': train_data_orig,
        'test_data_orig': test_data_orig,
        'train_data_ml': train_data_ml,
        'test_data_ml': test_data_ml,
        # 'valid': valid,
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
        'n_features_orig': n_features_orig,
        'n_features_median': n_features_median,
        'n_obs': n_obs,
        'n_obs_median': n_obs_median,
        # 'valid_X': valid_X,
        # 'valid_y': valid_y,
        'train_y_recon': train_y_recon,
        'train_y_extrapol': train_y_extrapol,
        'test_y_recon': test_y_recon,
        'test_y_extrapol': test_y_extrapol
    }
    
    return dataload

def get_median(array, axis = 1):
    # https://numpy.org/doc/stable/reference/generated/numpy.median.html
    return np.median(array, axis = axis).reshape(data_size, 1)  #, keepdims=True)


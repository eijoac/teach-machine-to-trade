"""utility function collections"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_data(col='close'):
    """ Reads and processes S&P500 and T-bill data. Returns a 2 x T-1 array. """

    # get raw data: S&P500 daily price index, and T-Bill daily yield
    sp500 = pd.read_csv('data/daily_SPXTR.csv', usecols=[col])
    rf_daily_return = pd.read_csv('data/daily_RF.csv', usecols=[col])

    # get S&P500 daily returns
    sp_daily_return = (sp500 - sp500.shift()) / sp500

    # remove the first row, and return a 2 by T-1 numpy array
    return np.array([sp_daily_return[col].values[1:],
                     rf_daily_return[col].values[1:]])


def get_scaler(env):
    """ Takes a env and returns a standard scaler for its observation space """

    # estimate max portfolio value
    max_portfolio_value = env.init_portfolio_value * 5

    # simulate portfolio value and sp_share distributions
    random_portfolio_value = np.random.rand(1, env.n_step) * max_portfolio_value
    random_sp_share = np.random.rand(1, env.n_step)

    # S&P and t-bill input (with lags)
    sp_rf_ts = env.sp_rf_ts
    sp_rf_ts_stack = np.repeat(sp_rf_ts, [env.lag+1, env.lag+1], axis=0)

    # all data input
    data_input = np.concatenate(
        (random_portfolio_value, random_sp_share, sp_rf_ts_stack), axis=0)

    # fit a standard scaler
    scaler = StandardScaler()
    scaler.fit(np.transpose(data_input))

    return scaler


def get_scaler_minmax(env):
    """ Takes a env and returns a minmax scaler for its observation space """

    # estimate max portfolio value
    max_portfolio_value = env.init_portfolio_value * 5

    # simulate portfolio value and sp_share distributions
    random_portfolio_value = np.random.rand(1, env.n_step) * max_portfolio_value
    random_sp_share = np.random.rand(1, env.n_step)

    # S&P and t-bill input (with lags)
    sp_rf_ts = env.sp_rf_ts
    sp_rf_ts_stack = np.repeat(sp_rf_ts, [env.lag+1, env.lag+1], axis=0)

    # all data input
    data_input = np.concatenate(
        (random_portfolio_value, random_sp_share, sp_rf_ts_stack), axis=0)

    # fit a min max scaler
    scaler = MinMaxScaler()
    scaler.fit(np.transpose(data_input))

    return scaler


def maybe_make_dir(directory):
    """ create a directory """
    if not os.path.exists(directory):
        os.makedirs(directory)

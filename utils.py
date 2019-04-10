"""utility functions"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_data(lag=0, other_indicator=False):
    """ Reads and processes S&P500 and T-bill data. Returns an array with time series rows. """

    # Get the investable asset file
    asset = pd.read_csv('data/InvestableAsset.csv')

    # get S&P500 daily returns
    asset['SP500'] = pd.DataFrame(
        (asset['SP500'] - asset['SP500'].shift()) / asset['SP500'])

    # check if we want lag for investable asset
    # https://stackoverflow.com/questions/48818213/make-multiple-shifted-lagged-columns-in-pandas
    # if lag != 0:
    #     name = asset.columns.values
    #     for i in range(1, len(name)):
    #         lags = range(1, lag+1)
    #         asset = asset.assign(
    #             **{"{}_{}".format(name[i], t): asset[name[i]].shift(t) for t in lags})

    #     # merge it with the other indicator based on date
    #     asset = pd.merge(asset, other, on='timestamp', how='inner')

    #     # in case asset.assign() above doesn't add new columns to the right of the original columns
    #     # stack; put investable asset returns first, and their lags and other indicator next
    #     data = np.hstack((asset.loc[:, name[1:len(name)]].values,
    #                       asset.drop(columns=name).values))
    # else:
    #     # if no lags required, simply join the investable asset with other indicators
    #     asset = pd.merge(asset, other, on='timestamp', how='inner')
    #     name = asset.columns.values
    #     data = asset.loc[:, name[1:len(name)]].values

    # check if we want lag for investable asset
    # https://stackoverflow.com/questions/48818213/make-multiple-shifted-lagged-columns-in-pandas
    if lag != 0:
        name = asset.columns.values
        for i in range(1, len(name)):
            lags = range(1, lag+1)
            asset = asset.assign(
                **{"{}_{}".format(name[i], t): asset[name[i]].shift(t) for t in lags})

    # remove NaN row(s) coming from the daily return and the potential lag calculatioin
    asset = asset.iloc[lag+1:, :]

    # check if we want other indicators
    if other_indicator:
        other = pd.read_csv('data/other.csv')
        # merge asset with the other indicator based on timestamp
        asset = pd.merge(asset, other, on='timestamp', how='inner')

    # remove timestamp column; convert dataframe to array
    data = asset.loc[:, asset.columns != 'timestamp'].values

    # transpose and return an array with time series rows;
    # the rows at the top are the investable asset daily returns
    return np.transpose(data)


def get_scaler(env):
    """ Takes a env and returns a standard scaler for its observation space """

    # estimate max portfolio value
    max_portfolio_value = env.init_portfolio_value * 5

    # simulate portfolio value and sp_share distributions
    random_portfolio_value = np.random.rand(
        1, env.n_step) * max_portfolio_value
    random_sp_share = np.random.rand(1, env.n_step)

    # all data input
    data_input = np.concatenate(
        (random_portfolio_value, random_sp_share, env.data), axis=0)

    # fit a standard scaler
    scaler = StandardScaler()
    scaler.fit(np.transpose(data_input))

    return scaler


def get_scaler_minmax(env):
    """ Takes a env and returns a minmax scaler for its observation space """

    # estimate max portfolio value
    max_portfolio_value = env.init_portfolio_value * 5

    # simulate portfolio value and sp_share distributions
    random_portfolio_value = np.random.rand(
        1, env.n_step) * max_portfolio_value
    random_sp_share = np.random.rand(1, env.n_step)

    # all data input
    data_input = np.concatenate(
        (random_portfolio_value, random_sp_share, env.data), axis=0)

    # fit a min max scaler
    scaler = MinMaxScaler()
    scaler.fit(np.transpose(data_input))

    return scaler


def maybe_make_dir(directory):
    """ create a directory """
    if not os.path.exists(directory):
        os.makedirs(directory)

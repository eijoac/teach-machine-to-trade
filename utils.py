import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(col='close'):
  """ Returns a 2 x n_step array """
  sp = pd.read_csv('data/daily_SPXTR.csv', usecols=[col])
  rf = pd.read_csv('data/daily_RF.csv', usecols=[col])

  # get lag for S&P
  sp_daily_return = (sp - sp.shift()) / sp

  # remove the first row, and return a numpy array
  return np.array([sp_daily_return[col].values[1:],
                   rf[col].values[1:]])


def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  max_total_portfolio_value = env.init_total_portfolio_value * 5
  step = env.n_step

  random_portfolio_value = np.random.rand(1, step) * max_total_portfolio_value
  random_sp_share = np.random.rand(1, step)

  sp_rf_ts = env.sp_rf_ts

  data = np.concatenate((random_portfolio_value, random_sp_share, sp_rf_ts), axis=0)

  scaler = StandardScaler()
  scaler.fit(np.transpose(data))

  # low = [0] * (env.n_stock * 2 + 1)

  # high = []
  # max_price = env.stock_price_history.max(axis=1)
  # min_price = env.stock_price_history.min(axis=1)
  # max_cash = env.init_invest * 3 # 3 is a magic number...
  # max_stock_owned = max_cash // min_price
  # for i in max_stock_owned:
  #   high.append(i)
  # for i in max_price:
  #   high.append(i)
  # high.append(max_cash)

  # scaler = StandardScaler()
  # scaler.fit([low, high])

  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
    

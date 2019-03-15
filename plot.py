import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import get_data


init_total_portfolio_value = 10000

# timestamp = "201903151000"
# init_sp_share = 0

# timestamp = "201903141815"
# init_sp_share = 0.5

# timestamp = "201903151145"
# init_sp_share = 1

timestamp = "201903151318"
init_sp_share = 1

data = get_data()

# train fraction
train_fraction = 0.6

train_split_idx = int(data.shape[1] * train_fraction)
train_data = data[:, :train_split_idx]
test_data = data[:, train_split_idx:]

buy_hold_return_train = np.sum([init_sp_share, 1 - init_sp_share] * np.product(1 + train_data[:, 1:], axis=1))
buy_hold_return_test = np.sum([init_sp_share, 1 - init_sp_share] * np.product(1 + test_data[:, 1:], axis=1))

buy_hold_value_train = init_total_portfolio_value * buy_hold_return_train
buy_hold_value_test = init_total_portfolio_value * buy_hold_return_test

cum_return_test_ts = np.cumprod(1 + test_data[:, 1:], axis=1)
buy_hold_value_ts_all_sp = init_total_portfolio_value * cum_return_test_ts[0, ]
buy_hold_value_ts_all_tbill = init_total_portfolio_value * cum_return_test_ts[1, ]
buy_hold_value_ts = init_total_portfolio_value * (init_sp_share * cum_return_test_ts[0, ] + (1 - init_sp_share) * cum_return_test_ts[1, ])

with open("./portfolio_val/" + timestamp + "-train.p", 'rb') as f:
    val_train = pickle.load(f)

with open("./portfolio_val/" + timestamp + "-test.p", 'rb') as f:
    val_test = pickle.load(f)

with open("./portfolio_val/" + timestamp + "-test-value-daily.p", 'rb') as f:
    val_test_daily = pickle.load(f)

with open("./portfolio_val/" + timestamp + "-test-state-daily.p", 'rb') as f:
    state_test_daily = pickle.load(f)

print("train-test split index: ", train_split_idx)
print("\n")
print("train: mean final portfolio value: ", np.mean(val_train))
print("train: buy-n-hold final portfolio value: ", buy_hold_value_train)
print("\n")
print("test: final portfolio value: ", val_test[0])
print("test: buy-n-hold final portfolio value: ", buy_hold_value_ts[-1])
print("test: buy-n-hold final portfolio value: ", buy_hold_value_test)
print("\n")
print(state_test_daily[0:10])

fig1, ax1 = plt.subplots()
ax1.plot(val_train)
ax1.axhline(np.mean(val_train))
ax1.axhline(buy_hold_value_train, color='r')

# print(cum_return_test)
fig2, ax2 = plt.subplots()
ax2.plot(buy_hold_value_ts_all_sp, color='r', label="BnH S&P500")
ax2.plot(buy_hold_value_ts_all_tbill, color='g', label="BnH T-Bill (daily refresh)")
ax2.plot(buy_hold_value_ts, label="BnH mix (init S&P500 frac: " + str(init_sp_share) + ")" )
ax2.plot(val_test_daily, color='k', label="dql (init S&P500 frac: " + str(init_sp_share) + ")")
ax2.legend(loc='upper left')

plt.show()
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import get_data


init_portfolio_value = 10000


# timestamp = "201903151000"
# init_sp_share = 0

# timestamp = "201903141815"
# init_sp_share = 0.5

# timestamp = "201903151145"
# init_sp_share = 1

# timestamp = "201903151318"
# init_sp_share = 1

# timestamp = "201903201751"
# init_sp_share = 0.5

# timestamp = "201903220925"
# init_sp_share = 0.5
# lag = 0

# timestamp = "201903221549"
# init_sp_share = 0.5
# lag = 7

# timestamp = "201903221756"
# init_sp_share = 0.5
# lag = 7

# timestamp = "201903221804"
# init_sp_share = 0
# lag = 7

# timestamp = "201903261237"
# init_sp_share = 0
# lag = 7

# timestamp = "201903281330"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 50
# timestamp = "201903281606"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 100; episode = 5000; final epsilon = 0.1
# timestamp = "201903281749"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 100; episode = 5000; final epsilon = 0.1
# timestamp = "201903281751"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 100; episode = 5000; final epsilon = 0.1; reward value difference (not log)
# timestamp = "201903290949"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 100; episode = 20000; final epsilon = 0.1; reward -1, 0, 1
# timestamp = "201903291424"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 400; episode = 20000; final epsilon = 0.1; reward log
# timestamp = "201903291722"
# init_sp_share = 0
# lag = 30

# test2: trick 2 implemented; update_feq = 50; episode = 5000; final epsilon = 0.1; reward log; learning rate = 0.00025; Hidden NN 32 nodes & 2 layers
# timestamp = "201904011723"
# init_sp_share = 0
# lag = 30


# test3: trick 2 implemented; update_feq = 50; episode = 5000; final epsilon = 0.1; reward log; learning rate = 0.001; Hidden NN 100 nodes & 2 layers
# timestamp = "201904011733"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_feq = 50; episode = 5000; final epsilon = 0.1; reward log; learning rate = 0.00025; Hidden NN 100 nodes & 2 layers
# timestamp = "201904011734"
# init_sp_share = 0
# lag = 30

# trick 2 implemented; update_freq = 50; episode = 2000; final epsilon = 0.1; reward log; learning rate = 0.0001; Hidden NN 100 nodes & 3 layers
# timestamp = "201904021024"
# init_sp_share = 0
# lag = 30

# test2: trick 2 implemented; update_freq = 50; episode = 5000; final epsilon = 0.1; reward log; learning rate = 0.0001; Hidden NN 100 nodes & 3 layers;
# total value normalization * 100
# timestamp = "201904021748"
# init_sp_share = 0
# lag = 30


# test3: trick 2 implemented; update_freq = 50; episode = 5000; final epsilon = 0.01; reward log; learning rate = 0.0001; Hidden NN 100 nodes & 3 layers;
# total value normalization * 5
# timestamp = "201904021751"
# init_sp_share = 0
# lag = 30

# test3: trick 2 implemented; update_freq = 50; episode = 5000; min_epsilon = 0.1; reward log; learning rate = 0.00005; Hidden NN 100 nodes & 3 layers;
# total value normalization * 5
timestamp = "201904021755"
init_sp_share = 0
lag = 30

data = get_data()

# train fraction
train_fraction = 0.6

train_split_idx = int(data.shape[1] * train_fraction)
train_data = data[:, :train_split_idx]
test_data = data[:, train_split_idx:]

buy_hold_return_train = np.sum([init_sp_share, 1 - init_sp_share] * np.product(1 + train_data[:, (1 + lag):], axis=1))
buy_hold_return_test = np.sum([init_sp_share, 1 - init_sp_share] * np.product(1 + test_data[:, (1 + lag):], axis=1))

buy_hold_all_sp_return_train = np.sum([1, 0] * np.product(1 + train_data[:, (1 + lag):], axis=1))
buy_hold_all_tbill_return_train = np.sum([0, 1] * np.product(1 + train_data[:, (1 + lag):], axis=1))

buy_hold_all_sp_value_train = init_portfolio_value * buy_hold_all_sp_return_train
buy_hold_all_tbill_value_train = init_portfolio_value * buy_hold_all_tbill_return_train

buy_hold_value_train = init_portfolio_value * buy_hold_return_train
buy_hold_value_test = init_portfolio_value * buy_hold_return_test

cum_return_test_ts = np.cumprod(1 + test_data[:, (1 + lag):], axis=1)
buy_hold_value_ts_all_sp = init_portfolio_value * cum_return_test_ts[0, ]
buy_hold_value_ts_all_tbill = init_portfolio_value * cum_return_test_ts[1, ]
buy_hold_value_ts = init_portfolio_value * (init_sp_share * cum_return_test_ts[0, ] + (1 - init_sp_share) * cum_return_test_ts[1, ])

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
print("train: buy-n-hold final portfolio value (all S&P500):", buy_hold_all_sp_value_train)
print("train: buy-n-hold final portfolio value (all T-bill):", buy_hold_all_tbill_value_train)
print("\n")
print("test: final portfolio value: ", val_test[0])
print("test: buy-n-hold final portfolio value: ", buy_hold_value_ts[-1])
print("test: buy-n-hold final portfolio value: ", buy_hold_value_test)
print("\n")
print(state_test_daily[0:20])
print(state_test_daily[400:460])

fig1, ax1 = plt.subplots()
ax1.plot(val_train)
ax1.axhline(np.mean(val_train), label="dql train average")
ax1.axhline(buy_hold_value_train, color='r', label="BnH mix (init S&P500 frac: " + str(init_sp_share) + ")" )
ax1.legend(loc='upper left')

# print(cum_return_test)
fig2, ax2 = plt.subplots()
ax2.plot(buy_hold_value_ts_all_sp, color='r', label="BnH S&P500")
ax2.plot(buy_hold_value_ts_all_tbill, color='g', label="BnH T-Bill (daily refresh)")
ax2.plot(buy_hold_value_ts, label="BnH mix (init S&P500 frac: " + str(init_sp_share) + ")" )
ax2.plot(val_test_daily, color='k', label="dql (init S&P500 frac: " + str(init_sp_share) + ")")
ax2.legend(loc='upper left')

fig3, ax3 = plt.subplots()
ax3.plot(np.stack(state_test_daily, axis=0)[:, 1], marker='o', markersize=1, linestyle='none')
plt.title("S&P500 holding fraction")

plt.show()


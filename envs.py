import gym
from gym import spaces
from gym.utils import seeding
# import numpy as np
# import itertools
import math


class TradingEnv(gym.Env):
    """
    A S&P500 and T-bill trading environment.

    State: [total value, % value in S&P500, S&P daily return, T-bill daily return]

    Action: -10% in S&P500 holding (0), hold (1), and +10% in S&P500 holding (2)
    """

    def __init__(self, train_data, init_sp_share=0.5):

        # data

        # jay: no round up
        # self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
        self.sp_rf_ts = train_data

        self.lag = 30

        # jay: keep the self.n_stock for now
        self.n_stock, self.n_step = self.sp_rf_ts.shape
        self.n_stock = self.n_stock - 1
        
        # instance attributes
        # self.init_invest = init_invest
        self.init_portfolio_value = 10000
        self.init_sp_share = init_sp_share
        self.cur_step = None
        self.total_portfolio_value = None
        # self.stock_owned = None
        # self.stock_price = None
        # self.cash_in_hand = None
        self.sp_share = None
        self.sp = None
        self.rf = None

        self.sp_lag = None
        self.rf_lag = None

        # action space
        self.action_space = spaces.Discrete(3)

        # observation space: give estimates in order to sample and build scaler
        # stock_max_price = self.stock_price_history.max(axis=1)

        # jay: modify to use Box (continuous) for observation space.
        # jay: in the old code observation_space is only used to assign state_size
        # jay: i keep this part to make it consistent with the old code; otherwise observation_space is not necessary

        # stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        # price_range = [[0, mx] for mx in stock_max_price]
        # cash_in_hand_range = [[0, init_invest * 2]]
        # self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

        # stock_range_max = [init_invest * 2 // mx for mx in stock_max_price]
        # price_range_max = [mx for mx in stock_max_price]
        # cash_in_hand_range_max = [init_invest * 2]
        # observation_space_max = stock_range_max + price_range_max + cash_in_hand_range_max
        # self.observation_space = spaces.Box(np.zeros(len(observation_space_max)), np.array(observation_space_max))

        # jay: for debug
        # print(price_range)
        # print(self.observation_space.shape)

        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = self.lag
        # self.stock_owned = [0] * self.n_stock
        # self.stock_price = self.stock_price_history[:, self.cur_step]
        # self.cash_in_hand = self.init_invest
        self.total_portfolio_value = self.init_portfolio_value
        self.sp_share = self.init_sp_share
        self.sp = self.sp_rf_ts[0, self.cur_step]
        self.rf = self.sp_rf_ts[1, self.cur_step]
        self.sp_lag = self.sp_rf_ts[0, self.cur_step - self.lag : self.cur_step]
        self.rf_lag = self.sp_rf_ts[1, self.cur_step - self.lag : self.cur_step]
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        # self.stock_price = self.stock_price_history[:, self.cur_step] # update price
        self.sp = self.sp_rf_ts[0, self.cur_step]
        self.rf = self.sp_rf_ts[1, self.cur_step]
        self.sp_lag = self.sp_rf_ts[0, self.cur_step - self.lag : self.cur_step]
        self.rf_lag = self.sp_rf_ts[1, self.cur_step - self.lag : self.cur_step]
        self._trade(action)
        cur_val = self._get_val()
        reward = math.log(cur_val) - math.log(prev_val)
        # reward = cur_val - prev_val
        # reward = math.copysign(1, cur_val - prev_val)
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        # obs.extend(self.stock_owned)
        # obs.extend(list(self.stock_price))
        # obs.append(self.cash_in_hand)
        obs.append(self.total_portfolio_value)
        obs.append(self.sp_share)
        obs.append(self.sp)
        obs.extend(self.sp_lag)
        obs.append(self.rf)
        obs.extend(self.rf_lag)
        return obs

    def _get_val(self):
        # return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
        return self.total_portfolio_value

    def _trade(self, action):
        # update S&P share based on action (at the end of a day)
        if action == 0:
            sp_share_tp = (self.sp_share - 0.1) if self.sp_share > 0.1 else 0
        elif action == 2:
            sp_share_tp = (self.sp_share + 0.1) if self.sp_share < 0.9 else 1
        else:
            sp_share_tp = self.sp_share

        # update total portfolio value based on next day's return
        sp_share_change_factor = sp_share_tp * (1 + self.sp)
        total_change_factor = sp_share_change_factor + \
            (1 - sp_share_tp) * (1 + self.rf)
        self.total_portfolio_value = self.total_portfolio_value * total_change_factor

        # update S&P share based on next day's return
        self.sp_share = sp_share_change_factor / total_change_factor

        # # all combo to sell(0), hold(1), or buy(2) stocks
        # action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        # action_vec = action_combo[action]

        # # one pass to get sell/buy index
        # sell_index = []
        # buy_index = []
        # for i, a in enumerate(action_vec):
        #   if a == 0:
        #     sell_index.append(i)
        #   elif a == 2:
        #     buy_index.append(i)

        # # two passes: sell first, then buy; might be naive in real-world settings
        # if sell_index:
        #   for i in sell_index:
        #     self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        #     self.stock_owned[i] = 0
        # if buy_index:
        #   can_buy = True
        #   while can_buy:
        #     for i in buy_index:
        #       if self.cash_in_hand > self.stock_price[i]:
        #         self.stock_owned[i] += 1 # buy one share
        #         self.cash_in_hand -= self.stock_price[i]
        #       else:
        #         can_buy = False

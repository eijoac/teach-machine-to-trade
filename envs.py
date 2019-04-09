"""Trading environment"""
import math

import gym
from gym import spaces
from gym.utils import seeding


class TradingEnv(gym.Env):
    """
    A S&P500 and T-bill trading environment.

    State: [portfolio value, fraction in S&P500, S&P daily return, T-bill daily return,
            and others (e.g. S&P and T-bill daily return lags)]

    Action: -10% in S&P500 holding (0), hold (1), and +10% in S&P500 holding (2)
    """

    def __init__(self, data, init_sp_share=0.5):

        # data
        self.data = data

        # instance attributes
        self.n_step = self.data.shape[1]
        self.init_portfolio_value = 10000
        self.init_sp_share = init_sp_share
        self.cur_step = None
        self.portfolio_value = None
        self.sp_share = None
        self.sp = None
        self.rf = None

        # state input other than porfolio value, sp share, sp & rf returns
        self.others = None

        # action space
        self.action_space = spaces.Discrete(3)

        # seed and start
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.cur_step = 0
        self.portfolio_value = self.init_portfolio_value
        self.sp_share = self.init_sp_share
        self.sp = self.data[0, self.cur_step]
        self.rf = self.data[1, self.cur_step]
        self.others = self.data[2:, self.cur_step]
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1

        # update price
        self.sp = self.data[0, self.cur_step]
        self.rf = self.data[1, self.cur_step]

        self.others = self.data[2:, self.cur_step]

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
        obs.append(self.portfolio_value)
        obs.append(self.sp_share)
        obs.append(self.sp)
        obs.append(self.rf)
        obs.extend(self.others)
        return obs

    def _get_val(self):
        return self.portfolio_value

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
        self.portfolio_value = self.portfolio_value * total_change_factor

        # update S&P share based on next day's return
        self.sp_share = sp_share_change_factor / total_change_factor

    def action_aug(self, action):
        """action augmentation; not the most efficient implementation"""
        prev_val = self._get_val()
        cur_step = self.cur_step + 1
        sp = self.data[0, cur_step]
        rf = self.data[1, cur_step]
        others = self.data[2:, cur_step]

        if action == 0:
            sp_share_tp = (self.sp_share - 0.1) if self.sp_share > 0.1 else 0
        elif action == 2:
            sp_share_tp = (self.sp_share + 0.1) if self.sp_share < 0.9 else 1
        else:
            sp_share_tp = self.sp_share

        # update total portfolio value based on next day's return
        sp_share_change_factor = sp_share_tp * (1 + sp)
        total_change_factor = sp_share_change_factor + \
            (1 - sp_share_tp) * (1 + rf)
        portfolio_value = self.portfolio_value * total_change_factor

        # update S&P share based on next day's return
        sp_share = sp_share_change_factor / total_change_factor

        # keep it this way in case not using log reward
        cur_val = portfolio_value
        # reward = math.log(total_change_factor)
        reward = math.log(cur_val) - math.log(prev_val)
        # reward = cur_val - prev_val
        # reward = math.copysign(1, cur_val - prev_val)

        done = cur_step == self.n_step - 1

        next_state = []
        next_state.append(portfolio_value)
        next_state.append(sp_share)
        next_state.append(sp)
        next_state.append(rf)
        next_state.extend(others)

        return next_state, reward, done

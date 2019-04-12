"""Trading with DRL"""
import argparse
import re

import pickle
import time
import numpy as np
import keras.backend as K

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=500,
                        help='number of episode to run; in test mode, the number is 1 regardless')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size for experience replay')
    parser.add_argument('-i', '--initial_sp_share', type=float, default=0.5,
                        help='initial fraction of S&P500 in total portfolio (0 to 1)')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    parser.add_argument('-w', '--weights', type=str,
                        help='a trained model weights')
    args = parser.parse_args()

    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    timestamp = time.strftime('%Y%m%d%H%M')

    lag = 30
    other_indicator = False

    # get data
    data = get_data(lag=lag, other_indicator=other_indicator)

    # train & test split
    train_fraction = 0.6

    train_split_idx = int(data.shape[1] * train_fraction)
    train_data = data[:, :train_split_idx]
    test_data = data[:, train_split_idx:]

    # print buy & hold returns in training and test
    # train_data[0:1, 1:]: 0:1 to get S&P500 and T-bill returns; 1: to remove first column as
    # we start from the end of the first day
    buy_hold_return_train = np.sum(
        [args.initial_sp_share, 1 - args.initial_sp_share] * np.product(1 + train_data[0:2, 1:], axis=1))
    buy_hold_return_test = np.sum(
        [args.initial_sp_share, 1 - args.initial_sp_share] * np.product(1 + test_data[0:2, 1:], axis=1))

    buy_hold_all_sp_return_train = np.sum(
        [1, 0] * np.product(1 + train_data[0:2, 1:], axis=1))
    buy_hold_all_tbill_return_train = np.sum(
        [0, 1] * np.product(1 + train_data[0:2, 1:], axis=1))

    buy_hold_all_sp_return_test = np.sum(
        [1, 0] * np.product(1 + test_data[0:2, 1:], axis=1))
    buy_hold_all_tbill_return_test = np.sum(
        [0, 1] * np.product(1 + test_data[0:2, 1:], axis=1))

    print("\n")
    print("initial S&P500 share:", args.initial_sp_share)
    print("train: buy & hold: ", buy_hold_return_train)
    print("test: buy & hold: ", buy_hold_return_test)

    print("train: buy & hold all S&P: ", buy_hold_all_sp_return_train)
    print("train: buy & hold all T-Bill: ", buy_hold_all_tbill_return_train)

    print("test: buy & hold all S&P: ", buy_hold_all_sp_return_test)
    print("test: buy & hold all T-Bill: ", buy_hold_all_tbill_return_test)

    # trading environment setup
    env = TradingEnv(train_data, args.initial_sp_share)

    # state_size is number of rows in data + 2 (portfolio value & sp share)
    state_size = data.shape[0] + 2

    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio_value = []

    # for recording test mode daily portfolio value and state
    test_portfolio_value_daily = []
    test_state_daily = []

    episode = args.episode

    if args.mode == 'test':
        # remake the env with test data
        env = TradingEnv(test_data, args.initial_sp_share)
        # load trained weights
        agent.load(args.weights)
        # when test, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{12}', args.weights)[0]
        episode_stamp = re.findall(r'(?<=-)\d+', args.weights)[0]

        # jay: no exploration in test mode
        agent.epsilon = -1
        # agent.epsilon = 0.3

        # jay: run 1 episode in test mode
        episode = 1

    for e in range(episode):
        state = env.reset()
        state = scaler.transform([state])
        agent.step = 1
        for time in range(env.n_step):
            action = agent.act(state)

            # action augmentation
            if args.mode == 'train':
                aug_actions = list(range(action_size))
                aug_actions.remove(action)
                for aug_action in aug_actions:
                    next_state, reward, done = env.action_aug(aug_action)
                    next_state = scaler.transform([next_state])
                    agent.remember(state, aug_action, reward, next_state, done)

            # taking optimum action
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])

            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)

            # record test mode daily state and portfolio value for plotting
            if args.mode == 'test':
                test_portfolio_value_daily.append(info['cur_val'])
                test_state_daily.append(scaler.inverse_transform(state)[0])

            state = next_state

            if done:
                print("episode: {}/{}, episode end value: {}".format(
                    e+1, episode, info['cur_val']))
                # append episode end portfolio value
                portfolio_value.append(info['cur_val'])
                break

            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)

        if args.mode == 'train' and (e + 1) % 500 == 0:  # checkpoint weights
            agent.save('weights/{}-{}-dqn.h5'.format(timestamp, e+1))

            # save final portfolio value training history to disk
            with open('portfolio_val/{}-{}-{}.p'.format(timestamp, e+1, args.mode), 'wb') as fp:
                pickle.dump(portfolio_value, fp)

    # leave it here for now in case we want to run more than one episodes in test later
    # save final portfolio value testing history to disk
    if args.mode == 'test':
        with open('portfolio_val/{}-{}-{}.p'.format(timestamp, episode_stamp, args.mode), 'wb') as fp:
            pickle.dump(portfolio_value, fp)

    # save test daily value and state
    if args.mode == 'test':
        with open('portfolio_val/{}-{}-{}-value-daily.p'.format(timestamp, episode_stamp, args.mode), 'wb') as fp:
            pickle.dump(test_portfolio_value_daily, fp)
        with open('portfolio_val/{}-{}-{}-state-daily.p'.format(timestamp, episode_stamp, args.mode), 'wb') as fp:
            pickle.dump(test_state_daily, fp)

    # save training parameters
    if args.mode == "train":
        with open('weights/{}-parameters.txt'.format(timestamp), 'w') as ft:
            print('initial investment: {}'.format(
                env.init_portfolio_value), file=ft)
            print('initial S&P500 fraction: {}\n'.format(
                env.init_sp_share), file=ft)
            print('number of state variables: {}'.format(state_size), file=ft)
            print('number of lags: {}'.format(lag), file=ft)
            print('other indicators: {}'.format(other_indicator), file=ft)
            print('number of actions: {}'.format(action_size), file=ft)
            print('number of episodes: {}\n'.format(args.episode), file=ft)
            print('agent: gamma: {}'.format(agent.gamma), file=ft)
            # hard code this one for now
            print('agent: epsilon start: {}'.format(1.0), file=ft)
            print('agent: epsilon decay rate: {}'.format(
                agent.epsilon_decay), file=ft)
            print('agent: epsilon minimum: {}'.format(
                agent.epsilon_min), file=ft)
            print('agent: memory size: {}'.format(
                agent.memory.maxlen), file=ft)
            print('agent: target network update frequency: {}\n'.format(
                agent.update_freq), file=ft)
            print('model summary:', file=ft)
            # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
            agent.model.summary(print_fn=lambda x: ft.write(x + '\n'))
            print('\nmodel: learning rate: {:f}'.format(
                K.eval(agent.model.optimizer.lr)), file=ft)

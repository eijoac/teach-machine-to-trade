import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=500,
                        help='number of episode to run; in test mode episode will be 1 regardless this value')
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

    # jay: no round up
    # data = np.around(get_data())
    data = get_data()

    # train fraction
    train_fraction = 0.6

    train_split_idx = int(data.shape[1] * train_fraction)
    train_data = data[:, :train_split_idx]
    test_data = data[:, train_split_idx:]

    buy_hold_return_train = np.sum(
        [args.initial_sp_share, 1 - args.initial_sp_share] * np.product(1 + train_data[:, 1:], axis=1))
    buy_hold_return_test = np.sum(
        [args.initial_sp_share, 1 - args.initial_sp_share] * np.product(1 + test_data[:, 1:], axis=1))

    print("buy hold train: ", buy_hold_return_train)
    print("buy hold test: ", buy_hold_return_test)

    env = TradingEnv(train_data, args.initial_sp_share)

    # jay: add for debug
    # print(env.observation_space.shape)

    # the only place observation_space is used
    # state_size = env.observation_space.shape[0]
    # hard code for now
    state_size = 4 + env.lag * 2
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio_value = []

    # for record test mode daily portfolio value for plotting
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

        # jay: no exploration in test mode
        agent.epsilon = -1
        # agent.epsilon = 0.3

        # run 1 episode in test mode
        episode = 1

    for e in range(episode):
        state = env._reset()
        state = scaler.transform([state])
        agent.step = 1
        for time in range(env.n_step - env.lag):
            action = agent.act(state)
            next_state, reward, done, info = env._step(action)
            next_state = scaler.transform([next_state])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            # record test mode daily state and portfolio value for plotting
            if args.mode == 'test':
                test_portfolio_value_daily.append(info['cur_val'])
                # print(scaler.inverse_transform(state)[0])
                test_state_daily.append(scaler.inverse_transform(state)[0])
            state = next_state
            if done:
                print("episode: {}/{}, episode end value: {}".format(
                    e + 1, args.episode, info['cur_val']))
                # append episode end portfolio value
                portfolio_value.append(info['cur_val'])
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e + 1) % 1000 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    if args.mode == 'test':
        with open('portfolio_val/{}-{}-value-daily.p'.format(timestamp, args.mode), 'wb') as fp:
            pickle.dump(test_portfolio_value_daily, fp)
        with open('portfolio_val/{}-{}-state-daily.p'.format(timestamp, args.mode), 'wb') as fp:
            pickle.dump(test_state_daily, fp)

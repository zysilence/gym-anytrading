from collections import defaultdict, Counter

from gym import spaces
import numpy as np
import pywt
import scipy.stats

from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv
from .trading_env import Actions, Positions


def process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    original = env.df.loc[:, ['Close', 'Open', 'High', 'Low']][start:end]

    log_diff = np.log(original) - np.log(original.shift(1))
    log_diff = log_diff.fillna(method='bfill').to_numpy()
    log_diff = log_diff.reshape((log_diff.shape[0], 1, log_diff.shape[1]))

    signal_features = log_diff

    return prices, signal_features


def get_observation(self):
    observation = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
    observation = observation.reshape((1, observation.shape[0], 1, observation.shape[2]))

    return observation


def calculate_reward(self, action):
    step_reward = 0

    current_price = self.prices[self._current_tick]
    prev_price = self.prices[self._current_tick - 1]
    price_diff = current_price / prev_price

    if action == Actions.Buy.value:
        step_reward += price_diff - 1
    elif action == Actions.Sell.value:
        step_reward += 1 - price_diff

    return step_reward


class MyStockEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
        self.trade_fee_bid_percent = 0  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        self.max_episode_length = 20

    _process_data = process_data
    _get_observation = get_observation
    _calculate_reward = calculate_reward



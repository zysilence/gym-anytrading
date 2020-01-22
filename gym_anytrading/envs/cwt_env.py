from collections import defaultdict, Counter

from gym import spaces
import numpy as np
import pywt
import scipy.stats

from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv
from .trading_env import Actions, Positions


def get_cwt_features(dataset, env):
    # TODO
    return dataset


def process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]

    original = env.df.loc[:, ['Close', 'Open', 'High', 'Low']][start:end]
    log_diff = np.log(original) - np.log(original.shift(1))
    log_diff = log_diff.fillna(method='bfill').to_numpy()
    signal_features = get_cwt_features(log_diff, env)

    return prices, signal_features


def get_observation(self):
    observation = self.signal_features[self._current_tick]

    return observation


def calculate_reward(self, action):
    step_reward = 0

    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price / last_trade_price

        if self._position == Positions.Long:
            step_reward += price_diff - 1

    return step_reward


class MyStockEnv(StocksEnv):
    _process_data = process_data
    _get_observation = get_observation
    _calculate_reward = calculate_reward


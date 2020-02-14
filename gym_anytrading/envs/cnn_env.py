""" Customized environment use CNN as feature extraction.
Author: sfan
"""

from collections import defaultdict, Counter

from gym import spaces
import matplotlib.pyplot as plt
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

    """
    signal_features = original.to_numpy()
    signal_features = signal_features.reshape((signal_features.shape[0], 1, signal_features.shape[1]))
    """

    return prices, signal_features


def get_observation(self):
    observation = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
    # observation = np.log(observation / observation[0][0])
    observation = observation.reshape((1, observation.shape[0], observation.shape[1], observation.shape[2]))

    # plt.cla()
    # plt.plot(observation[0, :, 0, 0])
    return observation


def calculate_reward_per_step(self, action):
    """
    Return reward every step.
    """
    step_reward = 0

    current_price = self.prices[self._current_tick]
    prev_price = self.prices[self._current_tick - 1]
    price_diff = current_price / prev_price

    if action == Actions.Buy.value:
        step_reward += price_diff - 1 - self.trade_fee_bid_percent - self.trade_fee_ask_percent
    elif action == Actions.Sell.value:
        step_reward += 1 - price_diff - self.trade_fee_bid_percent - self.trade_fee_ask_percent

    return step_reward


def calculate_reward_per_trade(self, action):
    """
    Return reward per trade, otherwise zero. The per trade reward is proportional to the per trade profit.
    This reward function is better than the per step one.
    """
    reward = 0
    trade = False

    if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade or self._done:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Positions.Long:
            reward = current_price / last_trade_price - 1 - self.trade_fee_bid_percent - self.trade_fee_ask_percent
    # debug
    if reward != 0:
        reward += 0
        reward += 0

    return reward


def update_profit_accumulated(self, action):
    """
    Use all the money in each trade.
    """
    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade or self._done:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Positions.Long:
            shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
            self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
        # [sfan] added
        if self._done is True:
            self._profit_history[self._history_idx % self._history_len] = self._total_profit


def update_profit(self, action):
    """
    Use fixed amount of money in each trade.
    """
    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade or self._done:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Positions.Long:
            self._total_profit += current_price / last_trade_price - 1 \
                                  - self.trade_fee_bid_percent - self.trade_fee_ask_percent
        # [sfan] added
        if self._done is True:
            self._profit_history[self._history_idx % self._history_len] = self._total_profit


class MyStockEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
        # No fee for debug
        self.trade_fee_bid_percent = 0  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        self.max_episode_length = 20

    _process_data = process_data
    _get_observation = get_observation
    _calculate_reward = calculate_reward_per_trade
    _update_profit = update_profit



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
    signal_features = get_dwt_features(log_diff, env)

    return prices, signal_features


def get_dwt_observation(self):
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


def get_dwt_features(dataset, env):
    waveletname = 'db5'
    dwt_features = []
    for signal_no in range(env.window_size, len(dataset)):
        features = []
        for signal_comp in range(0, dataset.shape[1]):
            signal = dataset[signal_no - env.window_size: signal_no, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            for coeff in list_coeff:
                features += get_features(coeff)
        dwt_features.append(features)
    X = np.array(dwt_features)
    paddings = np.zeros((env.window_size, X.shape[1]))
    X = np.row_stack((paddings, X))

    return X


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


class MyStockEnv(StocksEnv):
    _process_data = process_data
    _get_observation = get_dwt_observation
    _calculate_reward = calculate_reward


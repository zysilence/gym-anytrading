import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, MyStockEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, FeedForwardPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
import tensorflow as tf
import numpy as np


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def custom_cnn(scaled_images, **kwargs):
    """
    Customized CNN.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=(5, 1), stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=(4, 1), stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=(3, 1), stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))


class CustomCnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                              cnn_extractor=custom_cnn,
                                              feature_extraction="cnn", **_kwargs)


def callback(locals_, globals_):
    self_ = locals_['self']
    """
    # Log additional tensor
    if not self_.is_tb_set:
        with self_.graph.as_default():
            tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
            self_.summary = tf.summary.merge_all()
        self_.is_tb_set = True
    """
    # Log scalar value (here a random variable)
    total_reward = env._total_reward
    total_profit = env._total_profit
    summary = tf.Summary(value=[tf.Summary.Value(tag='total reward', simple_value=total_reward),
                                tf.Summary.Value(tag='total profit', simple_value=total_profit)])
    locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


if __name__ == '__main__':
    window_size = 120
    train_test_split = 0.8
    df_data = STOCKS_GOOGL

    total_bound = (window_size, len(df_data))
    train_bound = (window_size, len(df_data) * train_test_split)
    test_bound = (len(df_data) * train_test_split + 1, len(df_data))

    # Train
    env = MyStockEnv(df=df_data,
                     frame_bound=total_bound,
                     window_size=window_size)
    # env = gym.make('forex-v0', frame_bound=(10, len(FOREX_EURUSD_1H_ASK)), window_size=10)
    observation = env.reset()
    model = PPO2(CustomCnnPolicy, env, verbose=0, tensorboard_log="./tensorboard_log/")
    # model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=500000, tb_log_name="PPO2_Cnn", callback=callback)

    observation = env.reset()
    # Test
    while True:
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()
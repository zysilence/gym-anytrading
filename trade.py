import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, MyStockEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN


if __name__ == '__main__':
    window_size = 60
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
    model = PPO2(MlpPolicy, env, verbose=1)
    # model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    # Test
    while True:
        # action, _ = model.predict(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()

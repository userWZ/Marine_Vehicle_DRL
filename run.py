from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.pathFollowingEnv import PathFollowingEnv

env = PathFollowingEnv()
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_path_following")




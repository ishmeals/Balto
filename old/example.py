from stable_baselines3 import A2C
from balto import Balto

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


env = Balto(3, render_mode="human")

# model = A2C("CnnPolicy", env).learn(total_timesteps=1000)

# model = DQN("MlpPolicy", env, verbose=1)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)




def f(a=None):
    h = 1000
    obs, _ = env.reset(options={'a': a} if a else None)
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        # vec_env.render("human")
        h = min(h, info['cost'])
        if done: h
    return h


a = [15,13,8,
        14,17,9,16,
        10,18,0,4,12,
        3,7,2,5,
        11,6,1]

l = sorted([f() for _ in range(10)])
print(l)
print(min(l))

from stable_baselines3 import A2C
from balto import Balto

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


env = Balto(3, render_mode="human")

# model = A2C("CnnPolicy", env).learn(total_timesteps=1000)

# model = DQN("MlpPolicy", env, verbose=1)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# a = [15,13,8,
#         14,17,9,16,
#         10,18,0,4,12,
#         3,7,2,5,
#         11,6,1]

vec_env = model.get_env()
obs = vec_env.reset()

h = []
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    h.append(info[0]['cost'])
    # VecEnv resets automatically
    if done:
        print("done")
        break
        obs = vec_env.reset()

print(min(h), [i for i,x in enumerate(h) if x == min(h)])

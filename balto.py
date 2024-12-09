import gymnasium as gym
import numpy as np
from gymnasium import spaces
from functools import cache

@cache
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def flatten(g):
    return [x for row in g for x in row]

def encode(permutation):
    n = len(permutation)
    def parial_result(i):
        return sum(j < permutation[i] for j in permutation[i:]) * factorial(n-i-1)

    return sum(map(parial_result, range(n)))

def decode(n, lehmer):
    result = [(lehmer % factorial(n-i)) // factorial(n-1-i) for i in range(n)]
    used = [False] * n
    for i in range(n):
        counter = 0
        for j in range(n):
            if not used[j]:
                counter += 1
            if counter == result[i] + 1:
                result[i] = j
                used[j] = True
                break
    return result

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def done(self):
        for i, x in enumerate(flatten(self.G)):
            if i == self.N**2 and x != 0: return False
            elif i < self.N**2 and x != i+1: return False
            elif x != i+2: return False
        return True

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n):
        super().__init__()

        self.N = 3
        SIZE = 3*n**2-3*n+1

        self.G = [[15,13,8],
                [14,17,9,16],
                [10,18,0,4,12],
                [3,7,2,5],
                [11,6,1]]
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(12)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Discrete(factorial(SIZE))

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        assert seed is None and options is None
        return encode(flatten(self.G))

    def render(self):
        print(self.G)

    # def close(self):
    #     ...

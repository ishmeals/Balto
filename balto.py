import gymnasium as gym
import numpy as np
from gymnasium import spaces
from functools import cache
from math import ceil, log2

@cache
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def flatten(g):
    return [x for row in g for x in row]

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

def encode(permutation):
    n = len(permutation)
    def parial_result(i):
        return sum(j < permutation[i] for j in permutation[i:]) * factorial(n-i-1)

    return sum(map(parial_result, range(n)))

class Balto(gym.Env):
    """Custom Environment that follows gym interface."""

    steps = 0
    # hist = []

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def done(self):
        return self.steps >= 1000 or self.goal == [self.G[p] for p in self.order]

    def encode(self):
        i = encode([self.G[p] for p in self.order])
        return [int(x) for x in bin(i)[2:].zfill(self.outlen)]
        # return [self.G[p] for p in self.order]

    def __init__(self, n, render_mode="human"):
        self.render_mode = render_mode

        self.N = n
        self.SIZE = 3*n*(n-1)+1

        self.order = []
        l, r = 0, n-1
        for j in range(-n+1, n):
            for i in range(l, r+1):
                self.order.append((i, j))
            if j < 0: l -= 1
            else: r -= 1
        assert len(self.order) == self.SIZE

        self.goal = [*range(1, self.SIZE//2+1)] + [0] + [*range(self.SIZE//2+1, self.SIZE)]
        self.lookup = {i: p for (i, p) in zip(self.goal, self.order)}

        # a = [15,13,8,
        #         14,17,9,16,
        #         10,18,0,4,12,
        #         3,7,2,5,
        #         11,6,1]
        # self.G = {o: b for (o, b) in zip(self.order, a)}

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(12)
        self.outlen = int(ceil(log2(factorial(self.SIZE))))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.MultiDiscrete([2]* self.outlen)
        # self.observation_space = spaces.MultiDiscrete([self.SIZE]* self.SIZE)

    def cube_distance(self, a, b):
        return max(map(abs,(a[0]-b[0], a[1]-b[1], a[0]+a[1]-b[0]-b[1])))

    def cost(self):
        return sum(self.cube_distance(p, self.lookup[i]) for p,i in self.G.items() if i) + self.steps/1000000 + 1

    def step(self, action):
        self.steps += 1
        # self.hist.append(action)

        # assert set(self.G.values()) == set(range(self.SIZE))
        group = [p for p in self.G if self.G[p] == 0]
        if action % 6 == 0:
            group.append((group[0][0]+1, group[0][1]-1))
            group.append((group[0][0]+1, group[0][1]))
        elif action % 6 == 1:
            group.append((group[0][0]+1, group[0][1]))
            group.append((group[0][0], group[0][1]+1))
        elif action % 6 == 2:
            group.append((group[0][0], group[0][1]+1))
            group.append((group[0][0]-1, group[0][1]+1))
        elif action % 6 == 3:
            group.append((group[0][0]-1, group[0][1]+1))
            group.append((group[0][0]-1, group[0][1]))
        elif action % 6 == 4:
            group.append((group[0][0]-1, group[0][1]))
            group.append((group[0][0], group[0][1]-1))
        elif action % 6 == 5:
            group.append((group[0][0], group[0][1]-1))
            group.append((group[0][0]+1, group[0][1]-1))

        for i, p in enumerate(group):
            if p[0] < 0 and p[1] == self.N:
                group[i] = (p[0]+self.N, -self.N+1)
            elif p[0] > 0 and p[1] == -self.N:
                group[i] = (p[0]-self.N, self.N-1)
            elif sum(p) == self.N and p[1] > 0:
                group[i] = (p[0]-self.N+1, p[1]-self.N)
            elif sum(p) == -self.N and p[1] < 0:
                group[i] = (p[0]+self.N-1, p[1]+self.N)
            elif p[0] == self.N and p[1] > -self.N:
                group[i] = (-self.N+1, p[1]+self.N-1)
            elif p[0] == -self.N and p[1] < self.N:
                group[i] = (self.N-1, p[1]-self.N+1)

        if action < 6:
            self.G[group[0]] = self.G[group[2]]
            self.G[group[2]] = self.G[group[1]]
            self.G[group[1]] = 0
        else:
            self.G[group[0]] = self.G[group[1]]
            self.G[group[1]] = self.G[group[2]]
            self.G[group[2]] = 0

        d = self.c - self.cost()
        self.c = self.cost()
        return self.encode(), -self.c, self.done(), False, {'cost': self.c}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # assert seed is None and options is None

        if not options:
            # a = range(self.SIZE)
            a = np.random.permutation(self.SIZE)
        else:
            a = options['a']
            assert len(a) == self.SIZE

        self.G = {o: b for (o, b) in zip(self.order,a)}
        self.c = self.cost()

        return self.encode(), dict()

    def render(self):
        print(self.cost())
        # print(self.hist)

    # def close(self):
    #     ...

# env = Balto(3, render_mode="human")
# env.reset([2,6,1,9,17,0,5,16,10,13,18,7,12,4,15,3,14,8,11])
# env.step(0)
# env.step(9)
# env.step(4)
# print([env.G[p] for p in env.order])

# env = Balto(2, render_mode="human")
# env.reset([1,5,4,3,2,0,6])
# env.render()
# print([(i, env.cube_distance(p, env.lookup[i])) for p,i in env.G.items()])

# 61
# env = Balto(3, render_mode="human")
# env.reset([15,13,8,
#         14,17,9,16,
#         10,18,0,4,12,
#         3,7,2,5,
#         11,6,1])
# env.render()

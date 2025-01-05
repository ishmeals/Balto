import random
import numpy as np

N = 4
SIZE = 3*N*(N-1)+1
GOAL = np.concatenate((range(1, SIZE//2+1),[0], range(SIZE//2+1, SIZE)))

ORDER = []
l, r = 0, N-1
for j in range(-N+1, N):
    for i in range(l, r+1):
        ORDER.append((i, j))
    if j < 0: l -= 1
    else: r -= 1
assert len(ORDER) == SIZE

LOOKUP = {}
for i, p in enumerate(ORDER):
    LOOKUP[p] = i

class Balto:
    moves = [*range(12)]
    moves_inverse = [8, 9, 10, 11, 6, 7, 4, 5, 0, 1, 2, 3]

    def __init__(self): #, cw, a, steps=0):
        self.reset()

    def reset(self):
        self.state = GOAL.copy()
        self.zero = SIZE//2

    def done(self):
        return np.all(GOAL == self.state)

    def step(self, action):
        group = [ORDER[self.zero]]
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
            if p[0] < 0 and p[1] == N:
                group[i] = (p[0]+N, -N+1)
            elif p[0] > 0 and p[1] == -N:
                group[i] = (p[0]-N, N-1)
            elif sum(p) == N and p[1] > 0:
                group[i] = (p[0]-N+1, p[1]-N)
            elif sum(p) == -N and p[1] < 0:
                group[i] = (p[0]+N-1, p[1]+N)
            elif p[0] == N and p[1] > -N:
                group[i] = (-N+1, p[1]+N-1)
            elif p[0] == -N and p[1] < N:
                group[i] = (N-1, p[1]-N+1)

        if action < 6:
            self.state[LOOKUP[group[0]]] = self.state[LOOKUP[group[2]]]
            self.state[LOOKUP[group[2]]] = self.state[LOOKUP[group[1]]]
            self.state[LOOKUP[group[1]]] = 0
            self.zero = LOOKUP[group[1]]
        else:
            self.state[LOOKUP[group[0]]] = self.state[LOOKUP[group[1]]]
            self.state[LOOKUP[group[1]]] = self.state[LOOKUP[group[2]]]
            self.state[LOOKUP[group[2]]] = 0
            self.zero = LOOKUP[group[2]]

    def scrambler(self, scramble_length):
        while True:
            self.reset()
            cw = random.randint(0,1) == 0
            last = -1
            for _ in range(scramble_length):
                if cw: move = random.randint(0,5)
                else: move = random.randint(6,11)

                while last != -1 and move == self.moves_inverse[last]:
                    if cw: move = random.randint(0,5)
                    else: move = random.randint(6,11)

                self.step(move)

                yield self.state, move
                last = move
                cw = not cw

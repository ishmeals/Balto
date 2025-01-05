class SearchConfig:
    beam_width = 2**11                      # This controls the trade-off between time and optimality
    max_depth = 150
    ENABLE_FP16 = True                     # Set this to True if you want to solve faster

import torch
from contextlib import nullcontext
import time
from copy import deepcopy
import numpy as np

@torch.no_grad()
def beam_search(
        env,
        model,
        beam_width=SearchConfig.beam_width,
        max_depth=SearchConfig.max_depth,
        skip_redundant_moves=True,
    ):

    model.eval()
    with torch.autocast(str(device), dtype=torch.float16) if SearchConfig.ENABLE_FP16 else nullcontext():
        # metrics
        num_nodes_generated, time_0 = 0, time.time()
        candidates = [
            {"state":deepcopy(env.state), "path":[], "value":1.}
        ] # list of dictionaries

        for depth in range(max_depth+1):
            # TWO things at a time for every candidate: 1. check if solved & 2. add to batch_x
            batch_x = np.zeros((len(candidates), env.state.shape[-1]), dtype=np.int64)
            for i,c in enumerate(candidates):
                c_path, env.state = c["path"], c["state"]
                if c_path:
                    env.step(c_path[-1])
                    num_nodes_generated += 1
                    if env.done():
                        # Revert: array of indices => array of notations
                        c_path = [str(env.moves[i]) for i in c_path]
                        return True, {'solutions':c_path, "num_nodes_generated":num_nodes_generated, "times":time.time()-time_0}
                batch_x[i, :] = env.state

            # after checking the nodes expanded at the deepest
            if depth==max_depth:
                print("Solution not found.")
                return False, None

            # make predictions with the trained DNN
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_p = model(batch_x)
            batch_p = torch.nn.functional.softmax(batch_p, dim=-1)
            batch_p = batch_p.detach().cpu().numpy()

            # loop over candidates
            candidates_next_depth = []  # storage for the depth-level candidates storing (path, value, index).
            for i, c in enumerate(candidates):
                c_path = c["path"]
                value_distribution = batch_p[i, :] # output logits for the given state
                value_distribution *= c["value"] # multiply the cumulative probability so far of the expanded path

                for m, value in zip(env.moves_inverse, value_distribution): # iterate over all possible moves.
                    # predicted value to expand the path with the given move.

                    if c_path:
                        if (c_path[-1] < 6) == (m < 6): continue

                    if c_path and skip_redundant_moves:
                        # Two cancelling moves
                        if env.moves_inverse[c_path[-1]] == m:
                            continue

                    # add to the next-depth candidates unless 'continue'd.
                    candidates_next_depth.append({
                        'state':deepcopy(c['state']),
                        "path": c_path+[m],
                        "value":value,
                    })

            # sort potential paths by expected values and renew as 'candidates'
            candidates = sorted(candidates_next_depth, key=lambda item: -item['value'])
            # if the number of candidates exceed that of beam width 'beam_width'
            candidates = candidates[:beam_width]

env = Balto()
# env.state = np.asarray([15,13,8,
#         14,17,9,16,
#         10,18,0,4,12,
#         3,7,2,5,
#         11,6,1])
env.state = np.asarray([
15,35,24,23,
20,36,33,31,29,
27,26,30,25,18,2,
19,14,7,32,28,34,5,
4,13,0,8,1,11,
9,21,3,17,10,
12,22,6,16,
])

model = Model()
model.load_state_dict(torch.load('model.pth', weights_only=True))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
success, result = beam_search(env, model)
if success:
    print(result)
    s = list(map(int, result['solutions']))
    print(len(s))
    for n in s:
        if n < 6: print(n+1, end='')
        else: print(f'{4+n:x}', end='')
    print()

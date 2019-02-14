import numpy as np
from anytree import Node

S = 5
N=10
rollouts = [np.random.randint(2,size=(N,2)) for _ in range(S)]

def reward(rollout):
    return 2 # TODO

def make_tree(rollouts):
    root = {'value':0, 'count':0}
    for rollout in rollouts:
        rew = reward(rollout)
        current_dict = root
        for step in rollout:
            current_dict['value'] += rew
            current_dict['count'] += 1
            current_dict = current_dict.setdefault(step[0], {'value':0, 'count':0})
    return root


    
print(rollouts)
print(make_tree(rollouts))

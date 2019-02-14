import numpy as np

import rollout

rollout_num = 5
length_generator = lambda : 10
memory_size = 3

def create_rollouts():
    agent_a = rollout.defect_bot
    agent_b = rollout.tit_for_tat
    rollouts = rollout.multi_rollout(agent_a, agent_b, memory_size, length_generator, rollout_num)
    return rollouts


# rollouts = create_rollouts()

rollouts = [np.random.randint(2,size=(length_generator(), 2)) for _ in range(rollout_num)]


def reward(moves):
    return sum(rollout.moves_to_rewards(moves))


def make_tree(rollouts):
    root = {'value':0, 'count':0}
    for rollout in rollouts:
        rew = reward(rollout)
        current_dict = root
        for step in rollout:
            current_dict = current_dict.setdefault((step[0],step[1]), {'value':0, 'count':0})
            current_dict['value'] += rew
            current_dict['count'] += 1
    return root


# def collect_training_data(tree):
    


print(rollouts)
print(make_tree(rollouts))

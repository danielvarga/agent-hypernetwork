import numpy as np

import rollout
from rollout import DEF, COOP, NONE

rollout_num = 1000
length_generator = lambda : 5
memory_size = 3

def create_rollouts():
    agent_a = rollout.defect_bot
    agent_b = rollout.tit_for_tat
    agent_c = rollout.random_agent
    
    rollouts1 = rollout.multi_rollout(agent_c, agent_a, memory_size, length_generator, rollout_num)
    rollouts2 = rollout.multi_rollout(agent_c, agent_b, memory_size, length_generator, rollout_num)
    return rollouts1 + rollouts2


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

def get_children(tree):
    children = list(tree.keys())
    children.remove('value')
    children.remove('count')
    return children
                        
    

def collect_training_data(tree, memory_size):
    def collect_aux(tree, history):
        children = get_children(tree)
        # print(history, ": ", children)
        def_value = 0
        def_count = 0
        coop_value = 0
        coop_count = 0
        if (DEF, DEF) in children:
            def_value += tree[(DEF, DEF)]['value'][0]
            def_count += tree[(DEF, DEF)]['count']
        if (DEF, COOP) in children:
            def_value += tree[(DEF, COOP)]['value'][0]
            def_count += tree[(DEF, COOP)]['count']
        if (COOP, DEF) in children:
            coop_value += tree[(COOP, DEF)]['value'][0]
            coop_count += tree[(COOP, DEF)]['count']
        if (COOP, COOP) in children:
            coop_value += tree[(COOP, COOP)]['value'][0]
            coop_count += tree[(COOP, COOP)]['count']

        result_x = []
        result_y = []
        if coop_count > 0 and def_count > 0: # only extract training data when we have both cooperating and defecting successors            
            def_avg_value = def_value * 1.0 / def_count
            coop_avg_value = coop_value * 1.0 / coop_count
            result_x += [history[-memory_size:]]
            if coop_avg_value >= def_avg_value:
                result_y += [COOP]
            else:
                result_y += [DEF]
            
        for k in children:
            history2 = history + [k]
            result_x2, result_y2 = collect_aux(tree[k], history2)
            result_x += result_x2
            result_y += result_y2
        return result_x, result_y
    history = [(NONE, NONE) for _ in range(memory_size)]
    x, y = collect_aux(tree, history)
    return np.array(x), np.array(y)

rollouts = create_rollouts()
# print(rollouts)
tree = make_tree(rollouts)
# print(tree)
training_data = collect_training_data(tree, memory_size)
# for td in training_data:
#     print(td)

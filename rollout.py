import numpy as np
import keras

DEF, COOP, NONE = 0, 1, -1


# bigger is better
def reward(move_a, move_b):
    assert (move_a != NONE) and (move_b != NONE)
    if move_a == move_b:
        return (-1, -1) if move_a == COOP else (-2, -2)
    else:
        return (-3, 0) if move_a == COOP else (0, -3)


def moves_to_rewards(moves):
    return np.array([reward(move_a, move_b) for (move_a, move_b) in moves])

# a and b are agents.
# for concreteness, they get a kx2 array as input, and output a probability of cooperation.
# the array represents their past k interactions, later interactions with higher index.
# the elements of the array are DEF, COOP, NONE,
# where NONE means that the game hasn't started yet.
def rollout(a, b, k, n):
    state = np.ones((k, 2)) * NONE
    moves = []
    for i in range(n):
        coop_prob_a = a(state)
        coop_prob_b = b(state[:, ::-1]) # roles are inverted
        move_a = COOP if (coop_prob_a > np.random.uniform()) else DEF
        move_b = COOP if (coop_prob_b > np.random.uniform()) else DEF
        state[:-1, :] = state[1:, :]
        state[-1, 0] = move_a
        state[-1, 1] = move_b
        moves.append((move_a, move_b))
    return np.array(moves)


def multi_rollout(a, b, k, length_generator, rollout_num):
    return [rollout(a, b, k, length_generator()) for _ in range(rollout_num)]


def tit_for_tat(state):
    if state[-1, 1] == DEF:
        return 0.0
    else:
        return 1.0


def cooperate_bot(state):
    return 1.0


def defect_bot(state):
    return 0.0


def main():
    print("game between tit_for_tat and cooperate_bot:")
    moves = rollout(tit_for_tat, cooperate_bot, 3, 10)
    print(moves)
    print(moves_to_rewards(moves))
    print("game between tit_for_tat and defect_bot:")
    moves = rollout(tit_for_tat, defect_bot, 3, 10)
    print(moves)
    print(moves_to_rewards(moves))


if __name__ == "__main__":
    main()


"""
Implementation of small grid world example illustrated by David Silver
in his Reinforcement Learning Lecture3 - Planning by Dynamic Programming.
Author: zhs
Date: Sep 17, 2018

State space: 1-14 are non-terminal states, 0 and 15 are the terminal states
Action space: {north, west, south, east} for all non-terminal states
Transition probability: Actions leading out of the grid leave state unchanged, otherwise leading to the next state with
a probability of 100%
Rewards: −1 until the terminal state is reached
Policy: Agent follows uniform random policy, π(n|•) = π(e|•) = π(s|•) = π(w|•) = 0.25
Discount parameter: 1

"""
import numpy as np
from env_display import SmallGrid

GAMMA = 1.0  # discount parameter
ITERATION_STEP = 160  # iteration steps
states = np.arange(0, 16).reshape(4, 4)
d_actions = {'n': -4, 'w': -1, 's': 4, 'e': 1}


def get_index(one_state):
    index = np.argwhere(states == one_state)
    id_x = index[0][0]
    id_y = index[0][1]
    return id_x, id_y

class PolicyEva(object):
    """The class of policy evaluation method,
    update value function using Dynamic Programming"""

    def __init__(self):
        self.actions = ['n', 'w', 's', 'e']
        self.values = np.zeros((4, 4))

    def print(self):
        index = np.argwhere(states == 2)
        print(self.values)

    def next_move(self, cur_state, action):
        next_state = cur_state
        # move to next state except non-legal movement
        if cur_state in states[0] and action == self.actions[0] or cur_state in states[3] and action == self.actions[2]\
                or cur_state in states[:, 0] and action == self.actions[1] or cur_state in states[:, 3] and action == self.actions[3]:
            pass
        else:
            next_state = cur_state + d_actions[action]

        return next_state

    def get_successor(self, cur_state):
        successor = []
        # if current state in terminal state, return empty list
        if cur_state in [0, 15]:
            return successor

        for a in self.actions:
            next_state = self.next_move(cur_state, a)
            successor.append(next_state)

        return successor

    # update the value of one state
    def update_value(self, one_state):
        new_value = 0
        reward = 0 if one_state in [0, 15] else -1
        policy_prob = 1 / len(self.actions)

        successors = self.get_successor(one_state)
        for next_state in successors:
            # id_x, id_y = get_index(one_state)
            id_nx, id_ny = get_index(next_state)
            new_value += policy_prob*(reward + GAMMA*self.values[id_nx][id_ny])

        return new_value

    # update value function table
    def update_vtable(self):
        new_values = np.zeros((4, 4))
        for i in range(len(states[0])):
            for j in range(len(states)):
                new_values[i][j] = self.update_value(states[i][j])

        self.values = new_values



if __name__ == '__main__':
    s_g = SmallGrid()
    policy_eva = PolicyEva()

    policy_eva.print()
    for i in range(ITERATION_STEP):
         policy_eva.update_vtable()
         s_g.values = policy_eva.values
         # print(s_g.values)
         s_g.after(100, s_g.reset())

    s_g.mainloop()
    policy_eva.print()

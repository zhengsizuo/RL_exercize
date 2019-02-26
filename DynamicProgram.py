"""
Implementation of flappy birds using pygame module, in this game,
the goal is to find the female bird on the right top of the screen.
The algorithms includes Policy Iteration and Value Iteration.
Author: zhs
Date: Oct 30, 2018
"""
import pygame
import time
import numpy as np
from yuangyang_env import Bird, BgSet

# 设置屏幕为10*10=100个状态
width = 10
height = 10
# 两列障碍物的x坐标
first_col = 120  # 3*40
second_col = 280  # 7*40
# 绘制四列障碍物所需的参数
gap_distance = 90
col_one = 3
col_two = 4
col_three = 5
col_four = 2
# 设置初始状态，终止状态以及折扣参数
initial_state = 0
goal_state = 9
gamma = 0.99


class DynamicProgram(object):
    """动态规划类，包含策略迭代和值迭代两种方法"""
    def __init__(self):
        self.states = np.arange(0, 100)  # 状态空间，总共10*10=100个状态
        self.obstacle_states = []  # 障碍物状态列表
        self.edge_states = {}  # 边缘状态字典
        self.actions = ['n', 'w', 's', 'e']  # 定义动作空间
        self.d_actions = {'n': -10, 'w': -1, 's': 10, 'e': 1}  # 动作字典，键值表示状态转移
        self.values = np.zeros((100, 1))  # 值函数表
        self.evaluation_count = 100  # 每一轮策略评估中值函数更新的迭代次数
        self.theta = 1  # 迭代中判断收敛的小正数

        self._get_edge()
        self._get_obstacle()
        self.terminal_states = self.obstacle_states + [goal_state]  # 终止状态列表，包括障碍物状态和目标状态
        self._init_vtable()

    def _get_edge(self):
        """生成边缘状态字典"""
        for i in range(width):
            self.edge_states.setdefault('up', []).append(0+i)
            self.edge_states.setdefault('bottom', []).append(90+i)
        for j in range(height):
            self.edge_states.setdefault('left', []).append(0+j*width)
            self.edge_states.setdefault('right', []).append(9+j*width)

    def _get_obstacle(self):
        """生成障碍物状态列表"""
        for i in range(col_one):
            self.obstacle_states.append(3+i*width)
        for i in range(col_two):
            self.obstacle_states.append(63+i*width)
        for i in range(col_three):
            self.obstacle_states.append(7+i*width)
        for i in range(col_four):
            self.obstacle_states.append(87+i*width)

    def _init_vtable(self):
        for state in self.obstacle_states:
            self.values[state] = -100
        self.values[goal_state] = 100

    def reward_of(self, next_state):
        """转移到障碍物状态reward为-10，目标状态reward为+100，其余状态reward为-1"""
        if next_state in self.obstacle_states:
            return -10
        elif next_state == goal_state:
            return 100
        else:
            return -1

    def check_edges(self, cur_state, action):
        """检测边缘，小鸟的移动不能超出屏幕"""
        if cur_state in self.edge_states['up'] and action == 'n':
            return False
        if cur_state in self.edge_states['bottom'] and action == 's':
            return False
        if cur_state in self.edge_states['right'] and action == 'e':
            return False
        if cur_state in self.edge_states['left'] and action == 'w':
            return False

        return True

    def step(self, cur_state, action):
        """由当前状态和动作返回下一状态，回报以及是否为终止状态"""
        if cur_state in self.terminal_states:
            return cur_state, 0, True
        if not self.check_edges(cur_state, action):
            next_state = cur_state  # 如果状态要转移到屏幕外，则下一状态为当前状态
            r = -10  # 如果撞到屏幕，回报为-10
        else:
            next_state = cur_state + self.d_actions[action]
            r = self.reward_of(next_state)
        done = False
        if next_state in self.terminal_states:
            done = True

        return next_state, r, done

    def get_successor(self, cur_state):
        """从当前状态出发得到后继状态"""
        successors = []
        for a in self.actions:
            next_state, _, _ = self.step(cur_state, a)
            successors.append(next_state)
        return successors

    def update_q(self, one_state):
        """在策略评估中，更新任一状态的值函数"""
        new_value = 0
        pi_prob = 1 / len(self.actions)
        for a in self.actions:
            next_state, reward, done = self.step(one_state, a)
            new_value += pi_prob * (reward + gamma * self.values[next_state])  # 注意状态转移概率皆为1
        return new_value

    def policy_evaluation(self):
        """一轮策略评估(one sweep)"""
        for i in range(self.evaluation_count):
            old_value = sum(np.power(self.values, 2))
            new_value = np.zeros((100, 1))
            for state in self.states:
                new_value[state] = self.update_q(state)
            self.values = new_value.copy()
            delta_square = sum(np.power(self.values, 2)) - old_value
            if np.sqrt(abs(delta_square)) <= self.theta:
                # 若提前收敛，退出循环
                print("eval_count:{}".format(i))
                break

        for state in self.obstacle_states:
            self.values[state] = -100
        self.values[goal_state] = 100

    def gen_chain(self, cur_state):
        """根据贪婪策略生成马尔科夫链（状态序列）"""
        value_seq = []
        successors = self.get_successor(cur_state)
        for s in successors:
            value_seq.append(self.values[s])
        max_id = np.argmax(np.array(value_seq))
        return successors[max_id]

    def policy_improvement(self, cur_state):
        """提升当前状态到目标状态的策略, 返回策略序列"""
        state_seq = []
        start_state = cur_state
        while cur_state != goal_state:
            temp_state = self.gen_chain(cur_state)
            if temp_state in state_seq:
                break
            state_seq.append(temp_state)
            cur_state = temp_state
        state_seq.insert(0, start_state)
        return state_seq

    def policy_iteration(self, start_state):
        count = 0  # 记录策略迭代轮数
        old_policy = []
        while True:
            # 策略评估
            self.policy_evaluation()
            # 策略提升
            new_policy = self.policy_improvement(start_state)
            if new_policy == old_policy and new_policy[-1] == goal_state:
                break
            else:
                old_policy = new_policy
                print(old_policy)
                self.print_vtable()
                print('###################')
                count += 1
        print("The total time steps of policy iteration is: " + str(count))
        return new_policy

    def update_v(self, one_state):
        """在值迭代中，更新任一状态的值函数,取后续状态中的最大值"""
        new_value = []
        for a in self.actions:
            next_state, reward, done = self.step(one_state, a)
            # if next_state in self.obstacle_states:
            #     reward = -10
            new_value.append(reward + gamma * self.values[next_state])
        return max(new_value)

    def value_iteration(self, start_state):
        delta = 1
        count = 0
        while delta >= self.theta:
            # delta = 0
            count += 1
            old_value = sum(np.power(self.values, 2))
            new_value = np.zeros((100, 1))
            for state in self.states:
                new_value[state] = self.update_v(state)
            self.values = new_value.copy()
            delta_square = sum(np.power(self.values, 2)) - old_value
            delta = np.sqrt(abs(delta_square))
        print('The total time steps of value iteration is: '+str(count))

        cur_policy = self.policy_improvement(start_state)
        return cur_policy

    def print_vtable(self):
        """打印值函数表"""
        result = self.values.reshape(10, 10)
        result = np.around(result, decimals=1)
        for i in range(len(result)):
            for j in range(len(result[0])):
                print(result[i][j], '\t', end='')
                if (j + 1) % 10 == 0:
                    print(' ')


def run_env():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Find you")
    bird_male = Bird(screen)
    bird_female = Bird(screen)
    bird_female.rect.topleft = np.array([360, 0])
    bg_set = BgSet(screen)

    DP = DynamicProgram()
    DP.print_vtable()
    start = time.time()
    # cur_policy = DP.policy_iteration(initial_state)
    cur_policy = DP.value_iteration(initial_state)
    # 统计运行时间
    end = time.time()
    print('COST: {} s'.format(end - start))
    print('The ultimate value table converges to:')
    DP.print_vtable()

    while True:
        bg_set.blitme()
        if len(cur_policy) != 0:
            new_position = bird_male.state_to_coordinate(cur_policy.pop(0))
            bird_male.rect.topleft = new_position
        else:
            break
        bird_male.blitme()
        bird_female.blitme()

        pygame.display.update()
        time.sleep(0.5)


if __name__ == '__main__':
    run_env()

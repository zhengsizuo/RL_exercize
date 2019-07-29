"""
Implementation of flappy birds using pygame module, in this game,
the goal is to find the female bird on the right top of the screen.
The algorithms include  Sarsa, Q-learning，Expected Sarsa.
Author: zhs
Date: Nov 12, 2018
"""
import pygame
import time
import numpy as np
import pandas as pd
from yuangyang_env import Bird, BgSet

# 设置屏幕为10*10=100个状态
width = 10
height = 10
# 两列障碍物的x坐标
first_col = 120
second_col = 280
# 绘制四列障碍物所需的参数
gap_distance = 90
col_one = 3
col_two = 4
col_three = 5
col_four = 2
# 设置初始状态，终止状态以及折扣参数、学习率
initial_state = 0
goal_state = 9
gamma = 0.99
alpha = 0.4  # 使用期望Sarsa算法时可改为1，收敛更快


class TDLearning(object):
    """TD-learning类，包含sarsa、Q-learning"""
    def __init__(self):
        self.states = np.arange(0, 100)  # 定义状态空间，总共10*10=100个状态
        self.obstacle_states = []
        self.edge_states = {}
        self.actions = ['n', 'w', 's', 'e']
        self.d_actions = {'n': -10, 'w': -1, 's': 10, 'e': 1}  # 动作空间字典
        self.q_values = np.zeros([100, 4])  # q(s, a)
        self.q2_values = np.zeros([100, 4])
        self.policy_episode = []
        self.k = 10000  # 采样的样本数
        self.epsilon = 0.4

        self._get_edge()
        self._get_obstacle()
        self.terminal_states = self.obstacle_states + [goal_state]
        self._init_policy()

    def _get_edge(self):
        """生成边缘状态列表"""
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

    def turple2id(self, turple):
        """状态-动作对元组向值函数表坐标的转换"""
        x = turple[0]
        actions = np.array(self.actions.copy())
        y = np.argwhere(actions == turple[1])[0][0]
        return x, y

    def _init_policy(self):
        """初始化策略"""
        actions = self.actions.copy()
        for i in range(len(self.q_values)):
            a = np.random.permutation(actions)[0]
            self.policy_episode.append(a)
        self.policy_episode[goal_state-1] = 'e'
        self.policy_episode[goal_state+20] = 'n'

    def check_edges(self, cur_state, action):
        """如果撞到边缘，返回True"""
        if cur_state in self.edge_states['up'] and action == 'n':
            return True
        if cur_state in self.edge_states['bottom'] and action == 's':
            return True
        if cur_state in self.edge_states['right'] and action == 'e':
            return True
        if cur_state in self.edge_states['left'] and action == 'w':
            return True
        return False

    def step(self, cur_state, action):
        """根据当前状态和动作获得立即回报"""
        if cur_state in self.terminal_states:
            return cur_state, 0, True
        if self.check_edges(cur_state, action):
            next_state = cur_state
            r = -30  # 撞到边缘回报为-30
        else:
            next_state = cur_state + self.d_actions[action]
            if next_state in self.obstacle_states:
                r = -500  # 撞到障碍物回报为-250
            elif next_state == goal_state:
                r = 1000  # 找到终止状态回报为1000
            else:
                r = -1
        done = False
        if next_state in self.terminal_states:
            done = True
        return next_state, r, done

    def policy_behavior(self, s):
        """行为策略：使用epsilon-greedy策略产生动作"""
        actions = self.actions.copy()
        # q_values = self.q_values.copy()
        q_values = self.q_values.copy() + self.q2_values.copy()  # double q-learning算法中根据两个Q表值选取动作
        opt_id = np.argmax(q_values[s])
        a_opt = actions[opt_id]
        if np.random.random() > self.epsilon:
            a = a_opt
        else:
            actions.remove(a_opt)
            a = np.random.permutation(actions)[0]
        return a

    def sarsa_eval(self):
        """使用Sarsa算法评估策略"""
        for i in range(self.k):
            episode = []
            s = initial_state
            a = self.policy_behavior(s)
            while True:
                episode.append((s, a))
                s_, r, done = self.step(s, a)
                a_ = self.policy_behavior(s_)
                x, y = self.turple2id((s, a))
                next_x, next_y = self.turple2id((s_, a_))
                if s_ in self.terminal_states:
                    self.q_values[next_x][next_y] = 0
                td_target = r + gamma*self.q_values[next_x][next_y]
                self.q_values[x][y] += alpha * (td_target - self.q_values[x][y])

                s = s_
                a = a_
                if done:
                    break
                # 如果又回到之前经历过的状态
                if (s, a) in episode:
                    break

    def expected_sarsa(self):
        """期望Sarsa算法，将后续所有动作考虑进去"""
        for i in range(self.k):
            episode = []
            s = initial_state
            while True:
                a = self.policy_behavior(s)
                if (s, a) in episode:
                    break
                episode.append((s, a))
                s_, r, done = self.step(s, a)
                x, y = self.turple2id((s, a))
                ex_q = 0
                for a_ in self.actions:
                    next_x, next_y = self.turple2id((s_, a_))
                    if s_ in self.terminal_states:
                        self.q_values[next_x][next_y] = 0
                    ex_q += 1/4 * (self.q_values[next_x][next_y])

                td_target = r + gamma*ex_q
                self.q_values[x][y] += alpha * (td_target - self.q_values[x][y])
                s = s_

                if done:
                    break

    def q_learning(self):
        """使用Q-learning算法对策略进行评估"""
        for i in range(self.k):
            episode = []
            s = initial_state
            while True:
                a = self.policy_behavior(s)
                if (s, a) in episode:
                    break
                episode.append((s, a))
                s_, r, done = self.step(s, a)
                x, y = self.turple2id((s, a))
                q_max = np.max(self.q_values[s_])

                td_target = r + gamma * q_max
                episode.append((s, a, td_target))
                self.q_values[x][y] += alpha * (td_target - self.q_values[x][y])
                s = s_

                if done:
                    break

    def double_qlearning(self):
        """double q-learning算法：维护两个q表来更新"""
        for i in range(self.k):
            episode = []
            s = initial_state
            while True:
                a = self.policy_behavior(s)
                if (s, a) in episode:
                    break
                s_, r, done = self.step(s, a)
                episode.append((s, a))
                x, y = self.turple2id((s, a))
                if np.random.random() > 0.5:  # 类似抛硬币
                    a_id = np.argmax(self.q_values[s_])
                    a_ = self.actions[a_id]
                    next_x, next_y = self.turple2id((s_, a_))
                    td_target = r + gamma*self.q2_values[next_x][next_y]
                    self.q_values[x][y] += alpha * (td_target - self.q_values[x][y])
                else:
                    a_id = np.argmax(self.q2_values[s_])
                    a_ = self.actions[a_id]
                    next_x, next_y = self.turple2id((s_, a_))
                    td_target = r + gamma * self.q_values[next_x][next_y]
                    self.q2_values[x][y] += alpha * (td_target - self.q2_values[x][y])
                s = s_

                if done:
                    break

    def policy_improvement(self):
        """使用greedy策略改进，用于off-policy TD learning"""
        self.policy_episode = []
        q_values = self.q_values.copy()
        # q_values = self.q_values.copy() + self.q2_values.copy()  # double q-learning算法中根据两个Q表值选取动作
        for s_a in q_values:
            a_star = np.argmax(s_a)
            self.policy_episode.append(self.actions[a_star])

    def ep_policy_improvement(self):
        """使用epsilon-greedy策略改进，用于on-policy TD learning"""
        self.policy_episode = []
        q_values = self.q_values.copy()
        for s_a in q_values:
            actions = self.actions.copy()
            a_star = np.argmax(s_a)
            if np.random.random() > self.epsilon:
                self.policy_episode.append(self.actions[a_star])
            else:
                del actions[a_star]
                random_action = np.random.permutation(actions)[0]
                self.policy_episode.append(random_action)

    def find_policy(self, s):
        """找到一条从s状态到目标状态的马尔科夫链"""
        new_policy = [s]
        while True:
            s_, _, done = self.step(s, self.policy_episode[s])
            s = s_
            new_policy.append(s)
            if done:
                break
            if len(new_policy) > 100:
                # 防止在局部位置循环
                break

        return new_policy

    def policy_iteration(self, start_state):
        count = 0
        while True:
            # self.sarsa_eval()
            # self.ep_policy_improvement()
            # self.expected_sarsa()
            self.q_learning()
            # self.double_qlearning()
            self.policy_improvement()
            new_policy = self.find_policy(start_state)
            print(new_policy)
            if new_policy[-1] == goal_state:
                break
            count += 1
            if count % 10 == 0:
                # epsilon随时间衰减，表示探索减弱
                self.epsilon = self.epsilon / 2
                print('Current epsilon is:'+str(self.epsilon))
                self.print_info()

        print("The total time steps of policy iteration is: " + str(count))
        return new_policy

    def print_info(self):
        """打印Q表"""
        q_values = self.q_values.copy()
        q_output = pd.DataFrame(q_values, columns=['n', 'w', 's', 'e'])
        print(q_output)


def run_env():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Find you")
    bird_male = Bird(screen)
    bird_female = Bird(screen)
    bird_female.rect.topleft = np.array([360, 0])
    bg_set = BgSet(screen)

    start = time.time()
    td = TDLearning()
    new_policy = td.policy_iteration(initial_state)
    end = time.time()

    print('Final policy is: '+str(new_policy))
    print('COST: {} s'.format(end - start))
    print('The ultimate value table converges to:')
    td.print_info()

    s = initial_state
    bird_male.blitme()
    bird_female.blitme()
    while True:
        bg_set.blitme()
        s_, _, done = td.step(s, td.policy_episode[s])
        s = s_
        bird_male.rect.topleft = bird_male.state_to_coordinate(s)
        bird_male.blitme()
        bird_female.blitme()
        if done:
            break

        pygame.display.update()
        time.sleep(0.5)


if __name__ == '__main__':
    run_env()

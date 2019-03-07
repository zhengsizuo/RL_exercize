"""
Implementation of flappy birds using pygame module, in this game,
the goal is to find the female bird on the right top of the screen.
The algorithms includes Monte Carlo method with exploring starts and on-policy.
Author: zhs
Date: Nov 5, 2018
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
first_col = 120  # 3*40
second_col = 280  # 7*40
# 绘制四列障碍物所需的参数
gap_distance = 90
col_one = 3
col_two = 4
col_three = 5
col_four = 2
# 设置初始状态，终止状态以及折扣参数
initial_state = 0  # 使用on-policy方法时将初始状态设置到终止状态周围能收敛，如66和67等
goal_state = 9
gamma = 0.99


class MonteCarlo(object):
    """蒙特卡洛类，包含探索初始化和on-policy两种方法"""
    def __init__(self):
        self.states = np.arange(0, 100)  # 定义状态空间，总共10*10=100个状态
        self.obstacle_states = []
        self.edge_states = {}
        self.actions = ['n', 'w', 's', 'e']
        self.d_actions = {'n': -10, 'w': -1, 's': 10, 'e': 1}  # 动作空间字典，代表状态的变化
        self.q_values = np.zeros((100, 4))
        self.q_counts = np.zeros((100, 4))
        self.policy_episode = []
        self.k = 10000  # 采样的样本数
        self.epsilon = 0.2

        self._get_edge()
        self._get_obstacle()
        self.terminal_states = self.obstacle_states + [goal_state]

        # 在随机策略下对序列、状态-动作值函数表的初始化
        self._initial_qvalue()
        self.greedy_policy_improvement()
        # self.ep_policy_improvement()

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

    def _init_q_table(self):
        west_list = []  # 障碍物状态左边一个单位的状态
        east_list = []  # 障碍物状态右边一个单位的状态
        for item in self.obstacle_states:
            west_list.append((item-1, 'e'))
            east_list.append((item+1, 'w'))

        for s_a in west_list:
            x, y = self.turple2id(s_a)
            self.q_values[x][y] = -200
        for s_a in east_list:
            x, y = self.turple2id(s_a)
            self.q_values[x][y] = -200

        x1, y1 = self.turple2id((goal_state-1, 'e'))
        self.q_values[x1][y1] = 10000
        x2, y2 = self.turple2id((goal_state+10, 'n'))
        self.q_values[x2][y2] = 10000

        self.q2_values = self.q_values.copy()

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
            r = -20  # 撞到边缘回报为-20
        else:
            next_state = cur_state + self.d_actions[action]
            if next_state in self.obstacle_states:
                r = -200  # 撞到障碍物回报为-250
            elif next_state == goal_state:
                r = 10000  # 找到终止状态回报为1000
            else:
                r = -1
        done = False
        if next_state in self.terminal_states:
            done = True
        return next_state, r, done

    def turple2id(self, turple):
        """状态-动作对元组(s,'a')向值函数表坐标的转换"""
        x = turple[0]
        actions = np.array(self.actions)
        y = np.argwhere(actions == turple[1])[0][0]
        return x, y

    def gen_episode(self, s0, a0):
        """随机策略下生成一个序列样本并计算动作值函数"""
        actions = self.actions.copy()
        s, r, done = self.step(s0, a0)
        episode = [(s0, a0, r)]
        while True:
            if done:
                a_T = np.random.permutation(actions)[0]
                _, r, _ = self.step(s, a_T)
                episode.append((s, a_T, r))
                break
            a = np.random.permutation(actions)[0]
            s_, r, done = self.step(s, a)
            episode.append((s, a, r))
            s = s_
            # 如果序列长度超过200，退出循环
            if len(episode) > 200:
                break

        value = []
        return_val = 0
        for item in reversed(episode):
            # 反向计算累积回报
            return_val = return_val*gamma + item[2]
            value.append((item[0], item[1], return_val))

        # every-visit
        for item in reversed(value):
            x, y = self.turple2id((item[0], item[1]))  # 得到状态-动作对的位置
            # 递增式更新
            self.q_counts[x][y] += 1
            self.q_values[x][y] += (item[2] - self.q_values[x][y]) / self.q_counts[x][y]

    def _initial_qvalue(self):
        """初始化采样生成马尔科夫序列"""
        states = self.states.copy()
        actions = self.actions.copy()
        for i in range(self.k):
            s0 = np.random.permutation(states)[0]
            a0 = np.random.permutation(actions)[0]
            self.gen_episode(s0, a0)

    def gen_episode_pi(self, s0, a0):
        """根据贪心或epsilon-greedy策略生成序列进行新一轮策略评估"""
        s, r, done = self.step(s0, a0)
        episode = [(s0, a0, r)]
        while True:
            if done:
                _, r, _ = self.step(s, self.policy_episode[s])
                episode.append((s, self.policy_episode[s], r))
                break
            a = self.policy_episode[s]
            s_, r, done = self.step(s, a)
            episode.append((s, a, r))
            s = s_
            if len(episode) > 50:
                break

        return episode

    def compute_q(self, episode):
        """遍历序列中的状态-动作对，分为first-visit和every-first"""
        value = []
        return_val = 0
        for item in reversed(episode):
            return_val = return_val * gamma + item[2]
            value.append((item[0], item[1], return_val))

        # every-visit
        # for item in reversed(value):
        #     x, y = self.turple2id((item[0], item[1]))  # 得到状态-动作对的位置
        #     self.q_counts[x][y] += 1
        #     self.q_values[x][y] += (item[2] - self.q_values[x][y]) / self.q_counts[x][y]

        # first-visit
        visit_count = np.zeros_like(self.q_counts)
        for item in reversed(value):
            x, y = self.turple2id((item[0], item[1]))  # 得到状态-动作对的位置
            if visit_count[x][y] > 0:
                # 该状态-动作对被访问过
                pass
            self.q_counts[x][y] += 1
            self.q_values[x][y] += (item[2] - self.q_values[x][y]) / self.q_counts[x][y]
            visit_count[x][y] += 1

    def policy_evaluation(self):
        """MC方法中用探索初始化策略评估"""
        states = self.states.copy()
        actions = self.actions.copy()
        for i in range(self.k):
            s0 = np.random.permutation(states)[0]
            a0 = np.random.permutation(actions)[0]
            episode = self.gen_episode_pi(s0, a0)
            self.compute_q(episode)

    def on_policy_evaluation(self):
        """MC方法中without exploration starting策略评估"""
        for i in range(self.k):
            s0 = initial_state
            a0 = self.policy_episode[s0]
            episode = self.gen_episode_pi(s0, a0)
            self.compute_q(episode)

    def greedy_policy_improvement(self):
        self.policy_episode = []
        for s_a in self.q_values:
            action_id = np.argmax(s_a)
            self.policy_episode.append(self.actions[action_id])

    def ep_policy_improvement(self):
        """使用epsilon-greedy策略改进，用于on-policy MC"""
        self.policy_episode = []
        for s_a in self.q_values:
            actions = self.actions.copy()
            a_star = np.argmax(s_a)

            if np.random.random() > self.epsilon:
                # 选取最优动作
                self.policy_episode.append(self.actions[a_star])
            else:
                # 从其他动作中选取
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
            if len(new_policy) > 50:
                # 防止在局部位置循环
                break

        return new_policy

    def policy_iteraion(self, start_state):
        count = 0
        old_policy = []
        while True:
            self.policy_evaluation()
            self.greedy_policy_improvement()
            """注释部分为on-policy方法的策略评估和策略提升"""
            # self.on_policy_evaluation()
            # self.ep_policy_improvement()
            new_policy = self.find_policy(start_state)
            # 策略收敛的判断
            if new_policy == old_policy and new_policy[-1] == goal_state:
                break
            else:
                old_policy = new_policy
                print(old_policy)
                print('###################')
                count += 1
                if count % 5 == 0:
                    # epsilon随迭代次数衰减
                    self.epsilon = self.epsilon / 2
                    # print(self.epsilon)
        print("The total time steps of policy iteration is: " + str(count))

    def print_info(self):
        q_output = pd.DataFrame(self.q_values, columns=['n', 'w', 's', 'e'])
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
    mc = MonteCarlo()
    mc.policy_iteraion(initial_state)
    end = time.time()
    print('COST: {} s'.format(end - start))
    print('The ultimate value table converges to:')
    mc.print_info()

    s = initial_state
    bird_male.blitme()
    bird_female.blitme()
    while True:
        bg_set.blitme()
        s_, _, done = mc.step(s, mc.policy_episode[s])
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

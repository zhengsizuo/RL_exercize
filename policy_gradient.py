"""
Implementing policy gradient algorithm.
Author: zhs
Date: Dec 5, 2018
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np


class PolicyGradient(object):
    def __init__(self, n_input, n_output, n_hidden=10, learning_rate=0.01, reward_decay=0.95):
        self.n_input = n_input  # 状态空间
        self.n_output = n_output  # 动作空间
        self.n_hiddens = n_hidden  # 隐层数目
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []  # 元组(s, a, r)对应的序列
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        self.sess = tf.Session()
        self._init_input()
        self._build_net_work()
        self._init_op()

    def _init_input(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_input], name="state")
        self.a = tf.placeholder(tf.int32, [None, ], name="action")
        self.r = tf.placeholder(tf.float32, [None, ], name="reward")

    def _build_net_work(self):
        """全连接网络作为策略网络"""
        hidden = fully_connected(self.s, self.n_hiddens, activation_fn=tf.nn.relu,
                                 weights_initializer=self.initializer)
        self.logits = fully_connected(hidden, self.n_output, activation_fn=None,
                                 weights_initializer=self.initializer)
        self.action_prob = tf.nn.softmax(self.logits)

    def _init_op(self):
        action_one_hot = tf.one_hot(self.a, self.n_output)
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.logits, labels=action_one_hot)
        self.loss_function = tf.reduce_mean(neg_log_prob * self.r)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_function)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        """从最后一层softmax层输出的概率数组中选择动作"""
        state = state[np.newaxis, :]  # （n_inputs,)转化成(1, n_inputs)
        a_prob = self.sess.run(self.action_prob, feed_dict={self.s: state})
        # 按照概率选择动作, size=None时默认返回单个值
        action = np.random.choice(range(a_prob.shape[1]), size=None, p=a_prob.ravel())
        return action

    def discount_and_norm_rewards(self):
        """credit assignment 技巧，只考虑该动作之后的回报，并对回报进行归一化"""
        discounted_episode_rewards = np.zeros_like(self.ep_rewards)
        cumulative = 0
        for t in reversed(range(len(self.ep_rewards))):
            cumulative = cumulative * self.gamma + self.ep_rewards[t]
            discounted_episode_rewards[t] = cumulative

        # 将折扣后的回报Normalization
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def store_transition(self, s, a, r):
        self.ep_states.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def learn(self):
        """训练过程"""
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()
        self.sess.run([self.train_op, self.loss_function], feed_dict={
            self.s: self.ep_states,
            self.a: self.ep_actions,
            self.r: discounted_episode_rewards_norm
        })

        # 重置序列数据
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []


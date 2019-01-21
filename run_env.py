import gym
import numpy as np
import pandas as pd
from policy_gradient import PolicyGradient

env = gym.make('CartPole-v0')
env.reset()
env = env.unwrapped
# Policy gradient has high variance, seed for reproducibility
env.seed(1)
print("env.action_space", env.action_space.n)
print("env.observation_space", env.observation_space.shape[0])
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

RENDER_FLAG = False
EPISODES = 500  # 收集500条序列
MAX_STEP = 1500  # 每条序列最多1500步
rewards = []  # 记录每条序列回报的list


if __name__ == "__main__":
    PG = PolicyGradient(n_input=env.observation_space.shape[0], n_output=env.action_space.n)
    for episode in range(EPISODES):
        s = env.reset()
        for i in range(MAX_STEP):
            if RENDER_FLAG:
                env.render()
            # 与环境交互
            action = PG.choose_action(s)
            s_, reward, done, _ = env.step(action)
            PG.store_transition(s, action, reward)
            # 如果杆倒了或超出屏幕
            if done:
                ep_rewards_sum = np.sum(PG.ep_rewards)
                if ep_rewards_sum > 1000:
                    RENDER_FLAG = True
                else:
                    RENDER_FLAG = False
                rewards.append(ep_rewards_sum)
                PG.learn()
                break
            # 如果达到最大限制步数
            if i == (MAX_STEP-1):
                RENDER_FLAG = True
                rewards.append(i)
                PG.learn()
            s = s_

        if episode % 50 == 0:
            print("Episode {} ".format(episode))

    print(rewards)
    readout = pd.DataFrame(rewards)
    readout.to_csv('CartPole_Rewards.csv')

    filtered_rewards = []
    for i, ep_r in enumerate(rewards):
        try:
            filter_reward = np.mean([rewards[i], rewards[i+1], rewards[i+2], rewards[i+3], rewards[i+4]])
            filtered_rewards.append(filter_reward)
        except:
            break
    readout2 = pd.DataFrame(filtered_rewards)
    readout2.to_csv('CP_filter_rewards.csv')

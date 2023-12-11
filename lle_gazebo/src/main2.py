# -*- coding: utf-8 -*-
from importlib import reload


import LIP
import sys
import matplotlib.pyplot as plt
import numpy as np
import load_trajectory as ld
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from eppy.modeleditor import extendlist

import cartpole_pg
reload(cartpole_pg)
import signal
sys.path.append("./pyDmps")
sys.path.append(".")
import dmp_discrete
import net
reload(net)
reload(LIP)
record = []
#---------------------------------------------
def discount_rewards(reward, gamma):
    discounted_ep_rs = np.zeros_like(reward)
    running_add = 0
    for t in reversed(range(0, len(reward))):
        running_add = running_add * gamma + reward[t]
        discounted_ep_rs[t] = running_add
    return discounted_ep_rs

def norm_rewards(reward):
    reward -= np.mean(reward)
    reward /= (np.std(reward) + 0.0001)
    return reward



def cb(s, f):
    print ('recv signal', s)
    sys.exit(0)

action_set = {0:[-10.0,-10],
              1:[10,10],
              2:[10,-10],
              3:[10,0],
              4:[0,0],
              5:[0,10],
              6:[0,-10],
              7:[-10,0],
              8:[-10,10]}
# action_set = {0:[0.0,0.0],
#               1:[3.0, 0.0],
#               2:[-3.0, 0.0]}
obs, actions, rewards = [], [], []
batch_obs, batch_actions, batch_rewards = [], [], []
batch = 0
eval_param = 0

def reward_add(re, offset):
    l = len(re)
    for i in range(0, l):
        re[i] = re[i] + offset

def fun(lip):
    global i, external_force, observation, i_episode, agent
    global obs, actions, rewards, batch
    global batch_obs, batch_actions, batch_rewards, eval_param
    #S: x dx ddx y[0] y[1]
    #A: dmp.goal[0](-0.7 0.7) tau(0.02,10.0) external_force[0] external_force[1]
    #R: down -10 not down 1
    action = agent.act(observation,eval_param)
    # if i%20 ==0:
    # print(eval_param)
    external_force = np.array(action_set[action])
    y, dy, ddy = dmp.step(tau=tau_value, external_force=external_force)
    lip.set_swing_foot_pos(y[0], y[1])
    #将观测，动作和回报存储起来
    observation, reward, done, info = lip.update()
    # import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    if y[1] <= 0.0:
        if lip.stand_leg == 'left_leg':
            lip.switch_support_leg('right_leg')
            dmp.y0[0] = lip.left_foot_x
        else:
            lip.switch_support_leg('left_leg')
            dmp.y0[0] = lip.right_foot_x
        # print('dmp.y0[0]', dmp.y0[0])
        lip.update_orbital_energy()
        reward = np.log(1.0/((lip.E+0.065)**2+0.001))
        # reward = (-dmp.y0[0])*20.0 + lip.stand_leg_world_x*10.0
        # print('stand_leg' , lip.stand_leg_world_x)
        # print('E',lip.E)
        # print('rewards',reward,'len',(-dmp.y0[0])*10.0)
        dmp.goal[0] = 0.2
        dmp.reset_state()
    i = i + 1
    obs.append(observation)
    actions.append(action)
    rewards.append(reward)

    if done:
        batch = batch + 1
        dmp.goal[0] = 0.2
        dmp.y0[0] = 0.0
        dmp.reset_state()
        reward_add(rewards, lip.stand_leg_world_x * 2)
        # reward_add(rewards, reward)
        running_reward = np.sum(rewards)
        # running_reward = np.sum(batch_rewards)
        # print('X', lip.stand_leg_world_x*100.0)
        lip.reset(0.1, 0.1, 0.72, 0.001, 'left_leg')
        # i_episode = i_episode + 1
        # if eval_param == 1:
        #     print("episode:", i_episode, "rewards:", running_reward)
        # if eval_param == 1:
        #     eval_param = 0
        rewards = discount_rewards(rewards, 0.95)
        batch_actions.extend(actions)
        batch_obs.extend(obs)
        batch_rewards.extend(rewards)
        obs, actions, rewards = [], [], []
        if batch == 2:
            i_episode = i_episode + 1
            if eval_param == 1:
                print("episode:", i_episode, "rewards:", running_reward)
            if eval_param == 1:
                eval_param = 0
            batch_rewards = norm_rewards(batch_rewards)
            agent.train_step(batch_obs, batch_actions, batch_rewards)
            fig = plt.figure(5)
            ax = fig.add_subplot(111)
            ax.plot(batch_rewards)
            # ax.plot(rewards)
            # # ax.set_xlim([0,i_episode])
            # # ax.set_ylim([-10,batch_rewards])
            fig.show()
            # sys.exit()
            # plt.plot(i_episode , running_reward)
            # plt.show()
            batch_obs, batch_actions, batch_rewards = [], [], []
            batch = 0
            eval_param = 1
        i = 0

if __name__ == "__main__":
    i = 0
    external_force = np.array([0, 0])
    signal.signal(signal.SIGTERM, cb)
    signal.signal(signal.SIGINT, cb)
    trajectory = ld.loadTrajectory("swing_trajectory.txt")
    trajectory_x = trajectory[:, 0]
    trajectory_y = trajectory[:, 1]
    # plt.plot(trajectory_x, trajectory_y, 'r--', lw=2)
    dmp = dmp_discrete.DMPs_discrete(dmps=2, bfs=100)
    dmp.imitate_path(y_des=np.array([trajectory_x, trajectory_y]))
    dmp.goal[0] = 0.2
    dmp.y0[0] = 0.0
    dmp.reset_state()
    data_point_num = 300
    tau_value = (1 / dmp.dt)/ data_point_num
    hparams = {
            'input_size': 5,
            'hidden_size': 10,
            'num_actions': 3,
            'learning_rate': 0.01
    }
    # environment params
    eparams = {
            'num_batches': 40,
            'ep_per_batch': 10,
    }
    tf.Graph().as_default()
    sess = tf.Session()
    agent = cartpole_pg.PolicyGradientAgent(hparams, sess)
    sess.run(tf.global_variables_initializer())
    observation = np.array([0, 0, 0, 0, 0])
    i_episode = 0
    lip = LIP.LIP(0.1, 0.1, 0.77, 0.001, 'left_leg')
    # for i in range(100000):
    #     fun(lip)
    lip.inster_function = fun
    lip.run()


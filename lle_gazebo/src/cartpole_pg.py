# -*- coding: utf-8 -*-
from past.builtins import xrange

import gym
import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tf_slim.layers import layers as _layers
import pyglet


class PolicyGradientAgent(object):

    def __init__(self, hparams, sess):

        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                shape=[None, hparams['input_size']])

        hidden1 = _layers.fully_connected(
                inputs=self._input,
                num_outputs=hparams['hidden_size'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))

        hidden2 = _layers.fully_connected(
                inputs=hidden1,
                num_outputs=hparams['hidden_size'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
        #默认logists 是二维的
        logits = _layers.fully_connected(
                inputs=hidden2,
                num_outputs=hparams['num_actions'],
                activation_fn=None)

        # op to sample an action

        # get log probabilities
        self.act_gen_pro = tf.reshape(tf.nn.softmax(logits),[-1])

        log_prob = tf.log(tf.nn.softmax(logits))

        self._sample = tf.reshape(tf.multinomial(logits, 1), []) ###
        # training part of graph
        self._acts = tf.placeholder(tf.int32)
        self._advantages = tf.placeholder(tf.float32)

        # get log probs of actions from episode###
        #通过动作，推算出动作的索引
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
        ### 通过动作的索引寻找动作索引对应的概率
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss multiply
        print(act_prob)
        #a = raw_input()
        #a = input()
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))

        # update
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(loss)

    def act(self, observation,eval_param):
        # get one action, by sampling
        pro = self._s.run(self.act_gen_pro, feed_dict={self._input: [observation]})
        action1 = np.random.choice(np.arange(3), p=pro)
        action2 = np.random.choice(np.arange(5), p=np.ones(5)/ 5)
        action = np.random.choice(np.array([action1, action2]), p=np.array([0.5, 0.5]))
        if eval_param == 1:
            action = action1
        return action

    def train_step(self, obs, acts, advantages):
        batch_feed = { self._input: obs, \
                self._acts: acts, \
                self._advantages: advantages }
        self._s.run(self._train, feed_dict=batch_feed)


def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)


def main():

    env = gym.make('CartPole-v0')
    # monitor_dir = '/tmp/cartpole_exp1'
    # # env.monitor.start(monitor_dir, force=True)
    # env = gym.wrappers.Monitor(env, monitor_dir)

    # hyper parameters
    hparams = {
            'input_size': env.observation_space.shape[0],
            'hidden_size': 36,
            'num_actions': env.action_space.n,
            'learning_rate': 0.1
    }

    # environment params
    eparams = {
            'num_batches': 40,
            'ep_per_batch': 10,
    }

    with tf.Graph().as_default(), tf.Session() as sess:

        agent = PolicyGradientAgent(hparams, sess)

        sess.run(tf.initialize_all_variables())

        for batch in xrange(eparams['num_batches']):

            print ('=====\nBATCH {}\n===='.format(batch))

            b_obs, b_acts, b_rews = [], [], []

            for _ in xrange(eparams['ep_per_batch']):

                obs, acts, rews = policy_rollout(env, agent)

                print ('Episode steps: {}'.format(len(obs)))

                b_obs.extend(obs)
                b_acts.extend(acts)

                advantages = process_rewards(rews)
                b_rews.extend(advantages)

            # update policy
            # normalize rewards; don't divide by 0
            b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

            agent.train_step(b_obs, b_acts, b_rews)

        # env.monitor.close()


if __name__ == "__main__":
    main()

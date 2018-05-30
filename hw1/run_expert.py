#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt


def generate_rollout(env, expert_policy_file, max_timesteps, num_rollouts, render):

    max_steps = max_timesteps or env.spec.timestep_limit
    policy_fn = load_policy.load_policy(expert_policy_file)
    with tf.Session() as sess:
        tf_util.initialize()
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return observations, actions

def train(train_obs, train_acts, obs_space, act_space, epoch):
    train_inputs = tf.placeholder(tf.float32, shape=[1, obs_space])
    train_outputs = tf.placeholder(tf.float32, shape=[1, act_space])

    HIDDEN_UNITS = 64

    weights_1 = tf.get_variable(shape=[obs_space, HIDDEN_UNITS], initializer=tf.random_normal_initializer(mean=0, stddev=1), name="w1")
    bias_1 = tf.get_variable(shape=[1, HIDDEN_UNITS], initializer=tf.random_normal_initializer(mean=0, stddev=1), name="b1")

    layer_1_outputs = tf.nn.sigmoid(tf.matmul(train_inputs, weights_1) + bias_1)


    weights_2 = tf.get_variable(shape=[HIDDEN_UNITS, act_space], initializer=tf.random_normal_initializer(mean=0, stddev=1), name="w2")
    bias_2 = tf.get_variable(shape=[1, act_space], initializer=tf.random_normal_initializer(mean=0, stddev=1), name="b2")

    layer_2_outputs = tf.nn.sigmoid(tf.matmul(layer_1_outputs, weights_2) + bias_2)

    loss_function = 0.5 * tf.reduce_sum(tf.square(layer_2_outputs - train_outputs))

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_function)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        loss_list = []
        for i in range(epoch):
            for j in range(len(train_obs)):
                train_input = np.array(train_obs[j])
                train_input = np.reshape(train_input, (1, obs_space))
                #print(train_input)
                train_output = np.array(train_acts[j])
                train_output = np.reshape(train_output, (1, act_space))
                #print(train_output)
                _, loss = sess.run([train_step, loss_function], feed_dict={train_inputs: train_input, train_outputs: train_output})
                #print(loss)
            loss_list.append(loss)
        f = plt.figure()
        plt.plot(loss_list)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        #plt.show()
        f.savefig("warm_up.pdf", bbox_inches='tight')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    env = gym.make(args.envname)

    train_obs, train_acts = generate_rollout(env, args.expert_policy_file, args.max_timesteps, \
        args.num_rollouts, args.render)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    #print("obs_dim", obs_dim)
    #print("act_dim", act_dim)

    train(train_obs, train_acts, obs_dim, act_dim, args.epoch)

    # with tf.Session():
    #     tf_util.initialize()

    #     import gym
    #     env = gym.make(args.envname)
    #     max_steps = args.max_timesteps or env.spec.timestep_limit

    #     returns = []
    #     observations = []
    #     actions = []
    #     for i in range(args.num_rollouts):
    #         print('iter', i)
    #         obs = env.reset()
    #         done = False
    #         totalr = 0.
    #         steps = 0
    #         while not done:
    #             action = policy_fn(obs[None,:])
    #             observations.append(obs)
    #             actions.append(action)
    #             obs, r, done, _ = env.step(action)
    #             totalr += r
    #             steps += 1
    #             if args.render:
    #                 env.render()
    #             if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
    #             if steps >= max_steps:
    #                 break
    #         returns.append(totalr)

    #     print('returns', returns)
    #     print('mean return', np.mean(returns))
    #     print('std of return', np.std(returns))

    #     expert_data = {'observations': np.array(observations),
    #                    'actions': np.array(actions)}

if __name__ == '__main__':
    main()

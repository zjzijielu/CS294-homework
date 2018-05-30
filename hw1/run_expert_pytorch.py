#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert_pytorch.py experts/Humanoid-v1.pkl Humanoid-v2 --render \
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
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from model import *

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
                #print(type(action))
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

def variablesFromPair(pair, args):
	pair[0] = np.reshape(pair[0], (1, -1))
	pair[1] = np.reshape(pair[1], (1, -1))
	# get the target action index
	#target = pair[1].argmax(1)
	input_variable = Variable(torch.FloatTensor(pair[0]))
	target_variable = Variable(torch.FloatTensor(pair[1]))
	#print(target_variable)
	return (input_variable, target_variable)

def makePairs(obs, acts):
	pairs = []
	for i in range(len(obs)):
		pair = []
		pair.append(obs[i])
		pair.append(acts[i])
		pairs.append(pair)
	return pairs

def train(input_var, target_var, net, net_optimizer, criterion, args):
	loss = 0

	net_optimizer.zero_grad()

	#print(input_var)
	net_output = net(input_var)
	loss = criterion(net_output, target_var)
	loss.backward()

	net_optimizer.step()

	return loss.data[0]

def trainEpoch(net, pairs, args, test_pairs):
	n_epochs = args.epoch
	learning_rate = args.lr
	iter = 0
	net_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	criterion = nn.MSELoss()

	plot_losses = []
	plot_loss_total = 0

	for epoch in range(1, args.epoch+1):
		random.shuffle(pairs) 
		# converting pairs into variable
		training_pairs = [variablesFromPair(pair, args) for pair in pairs]

		for training_pair in training_pairs:
			iter += 1
			input_var = training_pair[0]
			target_var = training_pair[1]

			loss = train(input_var, target_var, net, net_optimizer, criterion, args)
			#print(loss)
			plot_loss_total += loss

			if iter % 500 == 0:
				plot_loss_avg = plot_loss_total / 500
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0

		print("epoch: %d, loss: %.6f, acc on test pairs: %.3f" % (epoch, plot_loss_avg, validate(net, test_pairs, args)))

	f = plt.figure()
	plt.plot(plot_losses)
	plt.ylabel('Loss')
	plt.xlabel('Iteration')
	f.savefig("result/%s.pdf" % args.envname, bbox_inches='tight')

def validate(net, pairs, args):
	valid_pairs = [variablesFromPair(pair, args) for pair in pairs]
	correcrt = 0
	for pair in valid_pairs:
		input_var = pair[0]
		target_var = pair[1]
		#print(target_var)

		output = net(input_var)
		#print(output)
		_, target_ind = torch.max(output, 1)
		_, output_ind = torch.max(output, 1)
		#print(output_ind)

		if torch.equal(target_ind.data, output_ind.data):
			correcrt += 1

	return (correcrt/len(pairs) )


def test(env, expert_policy_file, net, max_timesteps, num_rollouts, render):
	max_steps = max_timesteps or env.spec.timestep_limit
	policy_fn = load_policy.load_policy(expert_policy_file)
	with tf.Session() as sess:
		tf_util.initialize()
		returns = []

		for i in range(args.num_rollouts):
			observations = []
			actions = []
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				print("at step", steps)
				expected_action = policy_fn(obs[None,:])
				action = net(Variable(torch.FloatTensor(obs)))
				action = action.data.numpy()
				action = np.reshape(action, (1,-1))

				#print("expected action: ", expected_action)
				#print("predicted action: ", action)
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

	return returns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

    args = parser.parse_args()

    env = gym.make(args.envname)

    obs, acts = generate_rollout(env, args.expert_policy_file, args.max_timesteps, \
        args.num_rollouts, args.render)
    num_pairs = len(obs)

    pairs = makePairs(obs, acts)
    train_pairs = pairs[:int(0.8 * num_pairs)]
    test_pairs = pairs[int(0.8 * num_pairs):]

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = FFNet(obs_dim, act_dim, args.hidden_size)
    #print("obs_dim", obs_dim)
    #print("act_dim", act_dim)

    trainEpoch(net, train_pairs, args, test_pairs)

    #obs1 = train_pairs[0][0]
    #act1 = net(Variable(torch.FloatTensor(obs1)))

    #expected_act1 = train_pairs[0][1]

    #print("target act1: ", expected_act1)
    #print("predicted act1: ", act1)

    #validate(net, test_pairs, args)

    print("####### After training #######")

    #print("acc on training pairs: %.3f" % validate(net, training_pairs, args))
    returns = test(env, args.expert_policy_file, net, args.max_timesteps, args.num_rollouts, args.render)

    result_file = open('result/result_%s.txt' % (args.envname), "w")
    result_file.write("##### training setting #####\n")
    result_file.write("num of rollouts: %d \n" % args.num_rollouts)
    result_file.write("num of epochs: %d \n" % args.epoch)
    result_file.write("NN hidden size: %d \n" % args.hidden_size)
    result_file.write("learning rate: %f \n" % args.lr)
    result_file.write("mean return: %.4f \n" % np.mean(returns))
    result_file.write("std of return: %.4f \n" % np.std(returns))
    result_file.close()


if __name__ == '__main__':
    main()

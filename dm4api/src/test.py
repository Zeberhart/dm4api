from dm4api import create_env, create_agent

import argparse
import pickle
import os
from tensorflow import compat
from gym.spaces import *
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.policy import *
from rl.memory import *
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

class StoreMetricsCallback(Callback):

    rewards = []

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        self.rewards.append(logs["episode_reward"])


def main():
    compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--agent', required=True, dest='agenttype',  type=str)
    parser.add_argument('-e', '--env', default='DefaultEnv', dest='envtype', type=str)
    parser.add_argument('-m', '--model', default='MultiInput', dest='modeltype', type=str)
    parser.add_argument('-s', '--scheme', dest='scheme', type=str, default='new_scheme')
    parser.add_argument('-a', '--api', dest='api', type=str, default='libssh')
    parser.add_argument('-v', '--version', dest='modelversion', type=str, default='')
    parser.add_argument('-d', '--datadir', dest='datadir', type=str, default=os.path.join("..", "data"))
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default="weights") 
    parser.add_argument('--episodes', dest='episodes', type=str, default=10) 
    parser.add_argument('--visualize', action="store_true") 
    args = parser.parse_args()

    datadir = args.datadir
    outdir = os.path.join(datadir, args.outdir)
    modeltype = args.modeltype
    envtype = args.envtype
    envtype = args.envtype
    api = args.api
    apidir = os.path.join(datadir, "apis", args.api)
    scheme = args.scheme
    schemedir = os.path.join(datadir, "schema", args.scheme)
    episodes = int(args.episodes)
    visualize = args.visualize
    
    agenttype = args.agenttype
    modelversion = args.modelversion
    modeldir = os.path.join(outdir,'{}_{}_{}'.format(args.modeltype, envtype, api), str(modelversion))

    # Loading the environment
    env = create_env(envtype, apidir = apidir, schemedir = schemedir)
    assert len(env.action_space.shape) == 1
    agent = create_agent(agenttype=agenttype, env=env, modeldir=modeldir, schemedir=schemedir)
    callback = StoreMetricsCallback()

    agent.test(env, nb_episodes=episodes, visualize=visualize, verbose=2, callbacks=[callback])

    print(sum(callback.rewards)/len(callback.rewards))

if __name__ == "__main__":
    main()
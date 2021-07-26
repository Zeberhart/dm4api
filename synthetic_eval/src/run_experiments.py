import sys
sys.path.append("../../dm4api/src")
from experiment_configs import get_config
from dm4api import create_env, create_agent

import itertools
import traceback
import argparse
import pickle
import os
import numpy as np
from collections import Counter, defaultdict

from tensorflow import compat
from gym.spaces import *
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.policy import *
from rl.memory import *
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate

class ResultsCallback(Callback):
    results = []
    successes = 0
    first_appearances = []
    success_lengths = []
    sugg_penalties = 0
    info_penalties = 0
    length_penalties = 0
    eli_kw_observed = []
    eli_query_observed = []
    hc_divergence = None

    def __init__(self):
        self.results = []
        self.differences = 0
        self.successes = 0
        self.first_appearances = []
        self.success_lengths = []
        self.turn_penalties = 0
        self.sugg_penalties = 0
        self.info_penalties = 0
        self.length_penalties = 0
        self.eli_kw_observed = []
        self.eli_query_observed = []

    def on_step_end(self, episode, logs={}):
        """Called at end of each episode"""
        if logs["info"]:
            if logs["info"]["successful"]: 
                self.successes += 1
                self.first_appearances.append(logs["info"]["first-appearance"])
                self.success_lengths.append(logs["info"]["turn-penalty"])
            self.turn_penalties -= logs["info"]["turn-penalty"]
            self.sugg_penalties += logs["info"]["sugg-all-penalty"]
            self.info_penalties += logs["info"]["info-all-penalty"]
            if(logs["info"]["eli-kw-used"]):
                self.eli_kw_observed.append(logs["info"]["sugg-all-penalty"] + logs["info"]["info-all-penalty"] - logs["info"]["turn-penalty"])
            if(logs["info"]["eli-query-used"]):
                self.eli_query_observed.append(logs["info"]["sugg-all-penalty"] + logs["info"]["info-all-penalty"] - logs["info"]["turn-penalty"])

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        self.results.append(logs["episode_reward"])

    def mean(self):
        return np.mean(self.results)

def parse_config(config_name):
    config_dict = get_config(config_name)
    thing = []
    for attr, vals in config_dict.items():
         if vals != None:
            if isinstance(vals, list):
                options = [[attr, val] for val in vals]
            else:
                options = [[attr, vals]]
            if options: thing.append(options)
    configs = list(itertools.product(*thing))
    configs = [{a:v for a, v in c} for c in configs]
    return configs


def main():

    # SEED = 1337
    SEED = "13370705P370"

    compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--datadir', dest='datadir', type=str, default=os.path.join("..","..","dm4api","data"))
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default="weights") 
    parser.add_argument('-r', '--resultsdir', dest='resultsdir', type=str, default=os.path.join("..","data","experiment_results")) 
    parser.add_argument('-s', '--scheme', dest='scheme', type=str, default='new_scheme')

    parser.add_argument('-e', '--env', dest='envtype', type=str, default='DefaultEnv')
    parser.add_argument('-a', '--api', dest='api', type=str)

    parser.add_argument('-c', '--config-list', dest='configs', nargs='+', default=[])
    parser.add_argument('-n', '--name', dest='name', default="")
    parser.add_argument('--ep', dest='episodes', type=str, default=500) 
    parser.add_argument('-v',  dest='visualize', action="store_true") 
    args = parser.parse_args()

    datadir = args.datadir
    outdir = os.path.join(datadir, args.outdir)
    resultsdir = args.resultsdir
    envtype = args.envtype
    api = args.api
    apidir = os.path.join(datadir, "apis", args.api)
    scheme = args.scheme
    schemedir = os.path.join(datadir, "schema", args.scheme)
    episodes = int(args.episodes)
    config_names = args.configs
    name=args.name
    _print=args.visualize

    results = {}

    # # Loading the environment
    env = create_env(envtype, apidir = apidir, schemedir = schemedir)
    assert len(env.action_space.shape) == 1

    #Load configs
    configs = {config_name: parse_config(config_name) for config_name in config_names}

    #Run the experiment
    for config_name, agent_configs in configs.items():
        agent_results = []
        for i,agent_config in enumerate(agent_configs):
            print("Testing config %i/%i"%(i, len(agent_configs)))
            env.reset_user_seed(SEED)
            agent = create_agent(env=env, schemedir=schemedir, **agent_config)
            callback = ResultsCallback()
            agent.test(env, nb_episodes=episodes, visualize=False, verbose=0, callbacks=[callback])
            agent_results.append((agent_config,callback))
        results[config_name] = agent_results

    # Write results
    if not name:
        name = "".join([config_name for config_name in config_names]+[api, str(episodes)])
    with open(os.path.join(resultsdir,name), "w") as out:
        for config_name, config_results in results.items():
            out.write(config_name+"\n")
            if _print: 
                print(config_name)
            for result in config_results:
                agent_config,result_object = result
                out.write(  str(result_object.successes/episodes) + "\t" +

                            str(result_object.mean()) + "\t" +

                            str(result_object.turn_penalties/episodes) + "\t" +  
                            str(result_object.info_penalties/episodes) + "\t" +  
                            str(result_object.sugg_penalties/episodes) + "\t" + 

                            str(np.mean(result_object.first_appearances)) + "\t" +
                            str(np.mean(result_object.success_lengths)) + "\t" +

                            str(len(result_object.eli_kw_observed)/episodes) + "\t" +  
                            str(np.mean(result_object.eli_kw_observed)) + "\t" +  

                            str(len(result_object.eli_query_observed)/episodes) + "\t" +  
                            str(np.mean(result_object.eli_query_observed)) + "\t" +  

                            str(agent_config)+"\n\n")

                if _print: 
                    print(agent_config)
                    print(result_object.mean())
                    print()

if __name__ == "__main__":
    try:
        main()
    except:
        print(traceback.format_exc())
    print("\a")


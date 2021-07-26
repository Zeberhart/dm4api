from .default_env import DefaultLearnedAgent
from .default_env import DefaultHCAgent
from .default_env import DefaultBaselineAgent
from tensorflow.keras.optimizers import Adam
import os

def create_agent(agenttype=None, env=None, modeldir=None, **kwargs):
    agent = None

    if agenttype == 'DefaultLearnedAgent':
        agent = DefaultLearnedAgent(modeldir=modeldir, **kwargs)
    elif agenttype == 'DefaultHCAgent':
        agent = DefaultHCAgent(env=env, **kwargs)
    elif agenttype == 'DefaultBaselineAgent':
        agent = DefaultBaselineAgent(env=env, **kwargs)
    else:
        print("{} is not a valid agent type".format(agenttype))
        exit(1)
        
    return agent
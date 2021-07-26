import argparse
import pickle
import random
import os
from tensorflow import compat
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from gym.spaces import *
from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import *
from rl.core import Agent

class DefaultBaselineAgent(Agent):

    processor = None 
    

    env = None 


    def __init__(self, env, **kwargs):
        self.env = env
        self.compiled = True

    def reset_states(self):
        pass

    def forward(self, observation):
        cf = self.env.current_function
        last_action = self.env.last_user_action["action"]
        if last_action == "eli-info-all" and cf!=None:
            return self.env.AGENT_TYPES_DICT["info-all"]
        elif last_action == "eli-info" and cf!=None:
            return self.env.AGENT_TYPES_DICT["info"]
        elif last_action == "eli-sugg":
            return self.env.AGENT_TYPES_DICT["sugg"]
        elif last_action == "change-page":
            return self.env.AGENT_TYPES_DICT["change-page"]
        else:
            return self.env.AGENT_TYPES_DICT["sugg-all"]
        return -1

    def backward(self, reward, terminal):
        pass

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError()

    @property
    def layers(self):
        raise NotImplementedError()



def main():
    compat.v1.disable_eager_execution()
    pass

if __name__ == "__main__":
    main()
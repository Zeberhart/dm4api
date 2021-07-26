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

class DefaultHCAgent(Agent):

    processor = None 
    
    ELI_QUERY_THRESHOLD = .01
    ELI_KEYWORD_THRESHOLD = .07
    SUGG_THRESHOLD = .07
    SUGG_INFO_THRESHOLD = .56

    env = None 

    can_ask_more = True

    def __init__(self, env, eli_query_threshold=None, eli_keyword_threshold=None, sugg_threshold=None, sugg_info_threshold=None, **kwargs):
        self.env = env
        self.compiled = True
        if eli_keyword_threshold: self.ELI_KEYWORD_THRESHOLD=eli_keyword_threshold
        if eli_query_threshold: self.ELI_QUERY_THRESHOLD=eli_query_threshold
        if sugg_threshold: self.SUGG_THRESHOLD=sugg_threshold
        if sugg_info_threshold: self.SUGG_INFO_THRESHOLD=sugg_info_threshold

    def reset_states(self):
        self.can_ask_more=True
        pass

    def forward(self, observation):
        last_action = self.env.last_user_action["action"]
        highest_score = self.env.getHighestScore()
        cf = self.env.current_function

        if last_action == "dont-know":
            self.can_ask_more = False
            return self.env.AGENT_TYPES_DICT["sugg-all"]
        else:
            self.can_ask_more = True
            if last_action == "change-page":
                return self.env.AGENT_TYPES_DICT["change-page"]

            elif last_action == "eli-sugg":
                # if highest_score < self.ELI_QUERY_THRESHOLD and self.can_ask_more:
                #     return self.env.AGENT_TYPES_DICT["eli-query"]
                # elif highest_score < self.ELI_KEYWORD_THRESHOLD and self.can_ask_more:
                #     return self.env.AGENT_TYPES_DICT["eli-kw"]
                # if highest_score>self.SUGG_INFO_THRESHOLD:
                #     return self.env.AGENT_TYPES_DICT["sugg-info-all"]
                # elif highest_score>self.SUGG_THRESHOLD:
                return self.env.AGENT_TYPES_DICT["sugg"]
                # else:
                #     return self.env.AGENT_TYPES_DICT["sugg-all"]

            elif last_action == "eli-sugg-all":
                return self.env.AGENT_TYPES_DICT["sugg-all"]

            elif last_action in ["provide-query", "reject-kws", "reject-functions"]:
                if highest_score < self.ELI_QUERY_THRESHOLD and self.can_ask_more:
                    return self.env.AGENT_TYPES_DICT["eli-query"]
                elif highest_score < self.ELI_KEYWORD_THRESHOLD and self.can_ask_more:
                    return self.env.AGENT_TYPES_DICT["eli-kw"]
                if highest_score>self.SUGG_INFO_THRESHOLD:
                    return self.env.AGENT_TYPES_DICT["sugg-info-all"]
                elif highest_score>self.SUGG_THRESHOLD:
                    return self.env.AGENT_TYPES_DICT["sugg"]
                else:
                    return self.env.AGENT_TYPES_DICT["sugg-all"]

            elif last_action == "provide-kw":
                if highest_score < self.ELI_QUERY_THRESHOLD and self.can_ask_more:
                    return self.env.AGENT_TYPES_DICT["eli-query"]
                if highest_score>self.SUGG_INFO_THRESHOLD:
                    return self.env.AGENT_TYPES_DICT["sugg-info-all"]
                elif highest_score>self.SUGG_THRESHOLD:
                    return self.env.AGENT_TYPES_DICT["sugg"]
                else:
                    return self.env.AGENT_TYPES_DICT["sugg-all"]


                    
            elif last_action == "eli-info":
                if cf!=None:
                    return self.env.AGENT_TYPES_DICT["info"]
                else:
                    if highest_score < self.ELI_QUERY_THRESHOLD and self.can_ask_more:
                        return self.env.AGENT_TYPES_DICT["eli-query"]
                    elif highest_score < self.ELI_KEYWORD_THRESHOLD and self.can_ask_more:
                        return self.env.AGENT_TYPES_DICT["eli-kw"]
                    if highest_score>self.SUGG_INFO_THRESHOLD:
                        return self.env.AGENT_TYPES_DICT["sugg-info-all"]
                    elif highest_score>self.SUGG_THRESHOLD:
                        return self.env.AGENT_TYPES_DICT["sugg"]
                    else:
                        return self.env.AGENT_TYPES_DICT["sugg-all"]
            elif last_action == "eli-info-all":
                if cf!=None:
                    return self.env.AGENT_TYPES_DICT["info-all"]
                else:
                    if highest_score < self.ELI_QUERY_THRESHOLD and self.can_ask_more:
                        return self.env.AGENT_TYPES_DICT["eli-query"]
                    elif highest_score < self.ELI_KEYWORD_THRESHOLD and self.can_ask_more:
                        return self.env.AGENT_TYPES_DICT["eli-kw"]
                    if highest_score>self.SUGG_INFO_THRESHOLD:
                        return self.env.AGENT_TYPES_DICT["sugg-info-all"]
                    elif highest_score>self.SUGG_THRESHOLD:
                        return self.env.AGENT_TYPES_DICT["sugg"]
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
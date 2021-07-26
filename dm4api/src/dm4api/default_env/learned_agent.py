import argparse
import pickle
import os
from tensorflow import compat
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from gym.spaces import *
from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import *

class DefaultLearnedAgent(DQNAgent):

    def __init__(self, modeldir, schemedir, **kwargs):
        config = pickle.load(open(os.path.join(modeldir, "..", "config.pkl"), 'rb'))
        model = Model().from_config(config["model"]["config"])
        model.summary()
        memory = globals()[config["memory"]["class_name"]](**config["memory"]["config"])
        policy = globals()[config["policy"]["class_name"]](**config["policy"]["config"])
        super().__init__(model=model, nb_actions=config["nb_actions"], memory=memory, policy=policy,
                            nb_steps_warmup=100, gamma=.99, target_model_update=1e-3,)
        self.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        self.load_weights(os.path.join(modeldir, "weights.h5f"))
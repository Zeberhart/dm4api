from dm4api import create_env
from models import create_model

import argparse
import pickle
import os
import traceback
from tensorflow import compat
from gym.spaces import *
from rl.agents.dqn import DQNAgent, NAFAgent
from rl.policy import *
from rl.memory import *
from rl.random import OrnsteinUhlenbeckProcess

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam


def main():
    compat.v1.disable_eager_execution()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--env',  dest='envtype', default='DefaultEnv', type=str)
    parser.add_argument('-m', '--model', dest='modeltype', default='MultiInput', type=str)
    parser.add_argument('-d', '--datadir', dest='datadir', type=str, default=os.path.join("..", "data"))
    parser.add_argument('-s', '--scheme', dest='scheme', type=str, default='new_scheme')
    parser.add_argument('-a', '--api', dest='api', type=str, default='libssh')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default="weights") 
    parser.add_argument('-ep', '--epochs', dest='epochs', nargs='+', type=str, default=[100000]) 
    parser.add_argument('-w', '--wlen', dest='wlen', type=str, default=1) 
    parser.add_argument('-l', '--load', dest='load', type=str, default="") 
    parser.add_argument('-n', '--name', dest='name', type=str, default="") 
    parser.add_argument('-test', action="store_true") 
    args = parser.parse_args()
    
    modeltype = args.modeltype
    envtype = args.envtype
    datadir = args.datadir
    load = args.load
    api = args.api
    apidir = os.path.join(datadir, "apis", args.api)
    scheme = args.scheme
    schemedir = os.path.join(datadir, "schema", args.scheme)
    outdir = os.path.join(datadir, args.outdir)
    epochs = sorted([int(e) for e in args.epochs])
    wlen = int(args.wlen)
    test=args.test
    name=args.name

    # Loading the environment
    env = create_env(envtype, apidir = apidir, schemedir = schemedir)
    env.set_training()
    assert len(env.action_space.shape) == 1
    nb_actions = env.nb_actions[0] 
    nb_obs = env.nb_obs
    ### Some model architectures require this additional information ->
    try:
        nb_items = env.nb_items
    except:
        nb_items = None
    try:
        nb_slots = env.nb_slots
    except:
        nb_slots = None

    # Set RL memory and policy
    ### Haven't really experimented with these.
    memory = SequentialMemory(limit=50000, window_length=wlen)
    policy = BoltzmannGumbelQPolicy()

    # Create RL Agent
    ### DQNAgent uses a single model, outputs discrete-value action
    model = create_model(modeltype, {"nb_obs": nb_obs, "nb_items":nb_items, "nb_slots": nb_slots, "nb_actions": nb_actions, "wlen": wlen})
    print(model.summary())
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, 
        nb_steps_warmup=100, enable_dueling_network=True,  gamma=.99, target_model_update=1e-3,)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    if load:
        agent.load_weights(load)

    # Train Agent
    last_epoch = 0
    config = agent.get_config()
    config['modeltype'] = modeltype
    config['envtype'] = envtype
    config['api'] = api
    config['scheme'] = scheme
    config['nb_obs'] = nb_obs
    config['nb_actions'] = nb_actions
    config['nb_items'] = nb_items
    config['nb_slots'] = nb_slots
    config['wlen'] = wlen
    
    if not os.path.exists(os.path.join(outdir,'{}_{}_{}'.format(modeltype, envtype, api))):
        os.mkdir(os.path.join(outdir,'{}_{}_{}'.format(modeltype, envtype, api)))
    pickle.dump(config, open(os.path.join(outdir,'{}_{}_{}'.format(modeltype, envtype, api), 'config.pkl'), 'wb'))

    for e in epochs:
        agent.fit(env, nb_steps=e-last_epoch, visualize=False, verbose=2)
        # Save Agent and test
        agent.save_weights(os.path.join(outdir,'{}_{}_{}'.format(modeltype, envtype, api), name+str(e), 'weights.h5f'), overwrite=True)
        if test:
            env.set_training(False)
            agent.test(env, nb_episodes=10, visualize=True, verbose=1,)
            env.set_training(True)
        last_epoch = e
        
if __name__ == "__main__":
    try:
        main()
    except:
        print(traceback.format_exc())
    print("\a")



Dialogue Management
=================

This repository contains the source code and data used to implement the dialogue manager for API search. 

## Quick Start
### Test best-performing policies
* `python3 test.py --agent DefaultLearnedAgent -v 5200000` 
* `python3 test.py --agent DefaultLearnedAgent -v 1200000 -a allegro` 

### Train and test a policy using the libssh dataset
* `python3 train.py --epochs 25000` 
* `python3 test.py --agent DefaultLearnedAgent --version 25000`

## Train RL policy
To train the reinforcement learning model, navigate to src/ and run `train.py`. It will output model weights in data/models.

`train.py` accepts the following optional parameters:

* `--env [env]`: defines the Dialogue Manager architecture and learning environment. Only one environment currently implemented. Options: ["DefaultEnv" (default)]
* `--model [model]`: the neural model to be trained. Options: ["MultiInput" (default), "VanillaDense"] 
* `--scheme [scheme]`: used to train bigram probabilities in the user simulator. Only one currently implemented, based on the "Apiza" data published in _"A Wizard of Oz Study Simulating API Usage Dialogues with a Virtual Assistant." Eberhart, Z., Bansal, A., & Mcmillan, C. (2020). IEEE Transactions on Software Engineering._ ["new_scheme" (default)] 
* `--api [api]`: The API dataset the policy should train on. Options: ["libssh" (default), "allegro"] 
* `--datadir [datadir]`: The directory where the DM should look for the API dataset and "scheme" information. Default: "../data"
* `--outdir [outdir]`: The directory in the "datadir" directory where the learned policy weights should be saved. Default: "weights" 
* `--epochs [epochs]`: The desired number of training steps, including save points (e.g., to train for 100000 steps, saving snapshots at 25000, 50000, and 75000, use `--epochs 25000 50000 75000 100000`) Default: 100000
* `--wlen [wlen]`: The window length for DQN training. Default: 1
* `--load [load]`: Path to a saved model to continue training. Default: [None]
* `--name [name]`: A name to be appended to the model output directory. Default: [None]
* `-test`: Flag indicating that the learned policy should be tested after saving the model.


## Test any policy
To test any policy, navigate to src/ and run `test.py`.

`test.py` requires the following parameters:

* `--agent [agent]`: The "agent" implementing the policy to be tested. Options: ["DefaultRLAgent", "DefaultBaselineAgent", "DefaultHCAgent"]  

and accepts the following optional parameters:

* `--env [env]`: defines the Dialogue Manager architecture and testing environment. Only one environment currently implemented. Options: ["DefaultEnv" (default)]
* `--model [model]`: If loading a learned policy, the trained model architecture.  Options: ["MultiInput" (default), "VanillaDense"] 
* `--scheme [scheme]`: used to train bigram probabilities in the user simulator. Only one currently implemented, based on the "Apiza" data published in _"A Wizard of Oz Study Simulating API Usage Dialogues with a Virtual Assistant." Eberhart, Z., Bansal, A., & Mcmillan, C. (2020). IEEE Transactions on Software Engineering._ ["new_scheme" (default)] 
* `--api [api]`: The API dataset the policy should test on. Options: ["libssh" (default), "allegro"] 
* `--datadir [datadir]`: The directory where the DM should look for the API dataset and "scheme" information. Default: "../data"
* `--outdir [outdir]`: The directory in the "datadir" directory where all learned policy weights are stored. Default: "weights" 
* `--version [version]`: Name of the directory in the {outdir} directory that contains weights to be used. Indicates which version of a learned policy to be tested.
* `--episodes [episodes]`: Number of dialogues to be simulated.
* `-test`: Flag indicating that the tests should be rendered (according to the {env}'s render method).


## Making changes
To experiment with different agents, you can add new classes to the `environment` folder that extend the KerasRL `Agent` class. Make sure to export additional agent classes in `__init__.py` and modify `agent.py` to include any new agent types.

To modify the broader dialogue management environment, add a new environment folder. At the minimum, the envionment must export a class extending the Agent class and a class extending the KerasRL `Env` class.

##NOTE: Action Spaces
The actions listed in the paper map onto different action names in the repository:

###User actions
* provideQuery -> eli-query
* provideKeyword -> eli-kw
* rejectKeywords -> rejects-kws
* rejectAPI -> reject-functions
* unsure -> dont-know
* elicitInfoAPI -> eli-info
* elicitInfoAllAPI -> eli-info-all
* elicitSuggAPI -> eli-sugg
* elicitListResults -> eli-sugg-all
* changePage -> change-page

###System actions
* requestQuery -> eli-query
* suggKeywords -> eli-kw
* suggAPI -> sugg
* suggInfoAllAPI -> sugg-info-all
* infoAPI -> info
* infoAllAPI -> info-all
* listResults -> sugg-all
* changePage -> change-page











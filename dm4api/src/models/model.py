from .multiinput import MultiInput
from .vanilladense import VanillaDense

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'VanillaDense':
        mdl = VanillaDense(config)
    elif modeltype == 'MultiInput':
        mdl = MultiInput(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
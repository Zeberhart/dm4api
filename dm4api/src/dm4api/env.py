from .default_env import DefaultEnv

def create_env(envtype, apidir=None, schemedir=None):
    env = None
    if envtype == 'DefaultEnv':
        env = DefaultEnv(schemedir, apidir)
    else:
        print("{} is not a valid env type".format(envtype))
        exit(1)
        
    return env
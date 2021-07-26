import numpy as np


def get_config(config_name):
    return CONFIGS[config_name]()

def search_agent():
    return({
        "agenttype":"DefaultBaselineAgent",
    })

def hc_agent_default():
    return({
        "agenttype":"DefaultHCAgent",
    })

def hc_agent_loose():
    return({
        "agenttype":"DefaultHCAgent",
        "eli_query_threshold": list(np.arange(0, .4, 0.1)),
        "eli_keyword_threshold": list(np.arange(0, .6, 0.1)),
        "sugg_threshold": list(np.arange(.1, 1, 0.1)),
        "sugg_info_threshold": list(np.arange(.1, 1, 0.1))
    })

def hc_agent_tighter():
    return({
        "agenttype":"DefaultHCAgent",
        "eli_query_threshold": list(np.arange(0, .2, 0.1)),
        "eli_keyword_threshold": list(np.arange(.1, .3, 0.1)),
        "sugg_threshold": list(np.arange(.1, .4, 0.1)),
        "sugg_info_threshold": list(np.arange(.1, 1, 0.1))
    })

def hc_agent_tightest():
    return({
        "agenttype":"DefaultHCAgent",
        "eli_query_threshold": list(np.arange(0.01, .04, 0.005)),
        "eli_keyword_threshold": list(np.arange(.05, .15, 0.02)),
        "sugg_threshold": list(np.arange(.05, .15, 0.02)),
        "sugg_info_threshold": list(np.arange(.4, .6, .04))
    })


def agent_min_libssh_config():
    return({
        "agenttype":"DefaultLearnedAgent",
        "modeldir":[
            "../../dm4api/data/weights/MultiInput_EBEnv_libssh/1000000",
        ]
    })

def agent_max_libssh_config():
    return({
        "agenttype":"DefaultLearnedAgent",
        "modeldir":[
            "../../dm4api/data/weights/MultiInput_EBEnv_libssh/5200000",
        ]
    })

def agent_all_libssh_config():
    return({
        "agenttype":"DefaultLearnedAgent",
        "modeldir":[
            "../../dm4api/data/weights/MultiInput_EBEnv_libssh/"+str(n) for n in range(100000,10000001, 100000)
        ]
    })

def agent_max_allegro_config():
    return({
        "agenttype":"DefaultLearnedAgent",
        "modeldir":[
            "../../dm4api/data/weights/MultiInput_EBEnv_allegro/1200000",
        ]
    })

def agent_all_allegro_config():
    return({
        "agenttype":"DefaultLearnedAgent",
        "modeldir":[
            "../../dm4api/data/weights/MultiInput_EBEnv_allegro/"+str(n) for n in range(100000,10000001, 100000)
        ]
    })





CONFIGS = {
    "Baseline":search_agent,

    "HC":hc_agent_default,
    "HCLoose":hc_agent_loose,
    "HCTighter":hc_agent_tighter,
    "HCTightest":hc_agent_tightest,

    "RLLibssh":agent_max_libssh_config,
    "RLLibsshMin":agent_min_libssh_config,
    "RLLibsshAll":agent_all_libssh_config,

    "RLAllegro":agent_max_allegro_config,
    "RLAllegroAll":agent_all_allegro_config,
}
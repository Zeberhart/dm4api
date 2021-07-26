from .usersim import UserSim
from .dataset import Dataset

import pickle
import os
import numpy as np
from collections import Counter, defaultdict
from rl.core import Env
from gym.spaces import *
from gym.spaces import utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix



### TODO
# Currently, BetterEnv isn't quite replicating the likely real behavior of a good VA. In particular, it can learn not to answer
# the user with an expected type of Dialogue Act -- e.g. it doesn't have to answer an eli-sugg with a sugg. In reality, the best
# way for it to work would be to always answer the user appropriately, and use RL to figure out when and what kind of EXTRA information
# to supply. Also, use RL to figure out certain decisions e.g. should it reply to an eli-sugg with sugg or sugg-all?

class DefaultEnv(Env):


    K=6         #Number of functions to display in search results
    N=6         #Number of keywords to suggest
    MAX_TURNS = 25

    USER_TYPES=['provide-query', 'provide-kw', "reject-kws", 
    "reject-functions", "change-page", 'eli-info', 'eli-info-all',
    'eli-sugg', 'eli-sugg-all',  'dont-know', 'END']
    USER_TYPES_DICT = {u_type: i for i, u_type in enumerate(USER_TYPES)}

    AGENT_TYPES=['eli-query', 'eli-kw', 'info','info-all','sugg',
    'sugg-all','sugg-info-all', "change-page", "START"]  
    AGENT_TYPES_DICT = {a_type: i for i, a_type in enumerate(AGENT_TYPES)}
    
    SYSTEM_OPENING_ACT = "START"

    current_turn = None
    current_function = None
    query = None
    query_vector = None
    keywords=None
    functions_rejected=None
    result_index = None
    list_current = None

    previous_position = None
    training = False
        
    def reset_user_seed(self, seed):
        self.user.reset_seed(seed)

    def set_training(self, training=True):
        self.training = training

    def __init__(self, schemedir, apidir, training=False):
        self.training = training
        annotations = pickle.load(open(os.path.join(schemedir,"annotations.pkl"), 'rb'))
        sequences = pickle.load(open(os.path.join(schemedir,"sequences.pkl"), 'rb'))
        self.api = pickle.load(open(os.path.join(apidir,"api.pkl"), 'rb'))
        self.dataset = Dataset(self.api)
        self.nb_items = self.dataset.getDatabaseSize()
        self.user = UserSim(self.dataset, sequences, annotations)
        self.buildObservationSpace()
        self.buildActionSpace()

    def buildObservationSpace(self):
        """
            Defines the environment's observation space by getting info from the user object
        """
        self.observation_space = Dict({
            "system_action": Discrete(len(self.AGENT_TYPES)), 
            "user_action": Discrete(len(self.USER_TYPES)), 
            "function_specified": Discrete(2),
            "dont_know": Discrete(2),
            # "command_ignored": Discrete(2),
            "turns": Discrete(self.MAX_TURNS+1),
            "results": Box(low=np.zeros(self.dataset.getDatabaseSize()), high=np.ones(self.dataset.getDatabaseSize())),
        })
        self.observation_space.shape = (flatdim(self.observation_space),)

    def buildActionSpace(self):
        """
            Defines the environment's action space by getting info from the user object
        """
        self.action_types = self.AGENT_TYPES
        self.action_space = Dict({
            "action": Discrete(len(self.AGENT_TYPES)), 
        })
        self.action_space.shape = (len(self.action_types),)

    def reset(self):
        """
            Resets the turns of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        # Set the initial state

        if self.training:
            self.previous_position = self.nb_items

        # Various conversation features to be reported
        self.first_appearance = -1
        self.sugg_penalty = 0
        self.info_penalty = 0
        self.eli_kw_observed = False
        self.eli_query_observed = False

        self.dataset.reset()
        self.user.reset()

        self.current_turn = 0
        self.current_function = None
        self.query = ""
        self.query_vector = np.ones(self.dataset.getVocabularySize())
        self.keywords={"provided":set(), "rejected":set()}
        self.functions_rejected=set()
        self.result_index = 0
        self.dont_know = False

        self.history={
            'system_action': {"action": self.SYSTEM_OPENING_ACT},
            'user_action': None
        }
        user_action = self.user.respond(self.history["system_action"])
        self.first_response = self.processUserAction(user_action)
        observation = self.generateObservation()
        return observation

    def interactive_reset(self):
        """
            Resets the turns of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        # Set the initial state

        self.dataset.reset()

        self.current_turn = 0
        self.current_function = None
        self.query = ""
        self.query_vector = np.ones(self.dataset.getVocabularySize())
        self.keywords={"provided":set(), "rejected":set()}
        self.functions_rejected=set()
        self.result_index = 0
        self.dont_know = False

        self.history={
            'system_action': {"action": self.SYSTEM_OPENING_ACT},
            'user_action': None
        }

    def step(self, action):
        """
            Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # if self.current_turn<self.MAX_TURNS-1:
            # self.current_turn += 1
        

        self.current_turn += 1
        system_action = self.parseAction(action)
        
        # Used for logging and evaluation
        self.updateMetaState(system_action)

        self.processSystemAction(system_action)

        reward = self.calculateReward()

        user_action = self.user.respond(system_action)
        self.processUserAction(user_action)
        observation = self.generateObservation()
        done = self.isDone()
        if done:
            info = {    "successful": self.user.goals["satisfied"], 
                        "first-appearance": self.first_appearance, 
                        "turn-penalty": self.current_turn,
                        "sugg-all-penalty":self.sugg_penalty,
                        "info-all-penalty": self.info_penalty,
                        "eli-kw-used": self.eli_kw_observed,
                        "eli-query-used": self.eli_query_observed,
                    }
        else:
            info = {}
        if self.training:
            if done and self.user.goals["satisfied"]: reward+=30
        return observation, reward, done, info

    def updateMetaState(self, action):
        if action["action"] == "eli-query":
            self.eli_query_observed = True

        if action["action"] == "eli-kw":
            self.eli_kw_observed = True

        if self.first_appearance==-1:
            if (("function" in action and self.user.constraints["target_function_named"] == action["function"]) or
                    ("list" in action and self.user.constraints["target_function_named"] in action["list"])):
                self.first_appearance = self.current_turn

    def parseAction(self, action):
        """
            Converts a discrete action back to the action space
        # Returns
            full_action (dict): The action as translated into the full action space
        """
        action = self.AGENT_TYPES[action]


        full_action = {}
        full_action["action"] = action
        if action == "eli-kw":
            keywords = self.dataset.getSuggestedKeywords()
            full_action["keywords"] = keywords[:self.N]
        elif action == "info" or action == "info-all":
            full_action["function"] = self.current_function

        elif action == "sugg" or action == "sugg-info-all":
            top_hit = self.dataset.getTopHits(1)
            if not top_hit:
                full_action["action"] = "eli-query"
            else:
                functions = self.dataset.getTopHits(1, self.result_index)
                if functions:
                    full_action["function"] = functions[0]
                else:
                    full_action["function"] = ""

            self.result_index += 1

        elif action == "sugg-all":
            full_action["list"] = self.dataset.getTopHits(self.K, self.result_index)

        elif action == "change-page":
            self.result_index += self.K
            full_action["list"] = self.dataset.getTopHits(self.K, self.result_index)
        return full_action

    def processSystemAction(self, action):
        '''
            Update the environment's state in response to the system's action

        '''
        self.history["system_action"] = action

        def suggAll():
            self.list_current = True

        def changePage():
            self.list_current = True

        switcher = {
            'sugg-all':suggAll,
            'change-page':changePage,
        }

        if action["action"] in switcher:
            switcher[action["action"]]()


    def wasCommandIgnored(self, user_action_type, system_action_type):
        if  user_action_type=="eli-info" and system_action_type != "info": 
            return True
        if user_action_type=="eli-info-all" and system_action_type!="info-all":
            return True
        if user_action_type=="change-page" and system_action_type!="change-page":
            return True
        if user_action_type=="eli-sugg" and system_action_type not in ["sugg", "sugg-info-all"]:
            return True
        if user_action_type=="eli-sugg-all" and system_action_type not in ["sugg-all"]:
            return True

    def processUserAction(self, user_action):
        """ 
            Update the environment's state in response to the user's action
        """
        self.history["user_action"] = user_action
        dialogue_act = user_action["action"]
        self.current_function = None
        self.dont_know = False


        def provideQuery():
            self.query = user_action["query"]
            self.query_vector = self.dataset.getVectorForQuery(self.query)
            self.dataset.updateResults(query = self.query)
            self.result_index=0
            self.list_current = False
            return user_action

        def provideKw():
            self.keywords["provided"].add(user_action["keyword"])
            self.keywords["rejected"].discard(user_action["keyword"])
            self.dataset.updateResults(keywords = self.keywords)
            self.list_current = False
            self.result_index=0
            return user_action

        def rejectKws():
            self.keywords["provided"].difference_update(user_action["keywords"])
            self.keywords["rejected"].update(user_action["keywords"])
            self.dataset.updateResults(keywords = self.keywords)
            self.list_current = False
            return user_action

        def rejectFunctions():
            self.functions_rejected.update(user_action["functions"])
            self.dataset.updateResults(not_functions = self.functions_rejected)
            self.list_current = False
            return user_action

        def eliSugg():
            return user_action

        def eliInfo():
            self.current_function = user_action["function"]
            return user_action

        def eliInfoAll():
            self.current_function = user_action["function"]
            return user_action

        def changePage():
            return user_action

        def dontKnow():
            self.dont_know = True


        switcher = {
            'provide-query':provideQuery,
            'provide-kw':provideKw,
            'reject-kws':rejectKws,
            'reject-functions':rejectFunctions,
            'eli-sugg':eliSugg,
            'eli-sugg-all':eliSugg,
            'eli-info':eliInfo,
            'eli-info-all':eliInfo,
            'change-page':changePage,
            'dont-know':dontKnow
        }

        if dialogue_act in switcher:
            return switcher[dialogue_act]()
        else: return user_action

    def generateObservation(self):
        """ 
            This function should send the state of the environment (i.e. the
            last system action, goals, and constraints) to the user
            and request a response.
        # Returns
            obeservation (object): flattened obserevation
        """
        observation = {
            "system_action": self.AGENT_TYPES_DICT[self.last_system_action["action"]],
            "user_action": self.USER_TYPES_DICT[self.last_user_action["action"]],
            "function_specified": 1 if self.current_function != None else 0,
            "dont_know": 1 if self.dont_know else 0,
            "turns": self.current_turn,
            "results": self.dataset.function_scores,
        }
        return flatten(self.observation_space, observation)

    def calculateReward(self, bonus=0):
        """ 
            Calculates the return for the current step. 
        """
        reward = -1 + bonus

        if ((self.last_system_action["action"]=="sugg-all" and self.last_user_action["action"]!="eli-sugg-all") or 
            ("change-page" in self.last_system_action["action"] and self.last_user_action["action"]!='change-page')):
            reward -= .3
            self.sugg_penalty-=.3

        if "info-all" in self.last_system_action["action"] and "info-all" not in self.last_user_action["action"]:
            reward -= .5
            self.info_penalty -=.5

        if self.training:
            new_position = self.dataset.getPosition(self.user.constraints['target_function'])
            if new_position < self.previous_position:
                reward += 5*((self.previous_position-new_position)/self.nb_items)
                self.previous_position = new_position

            elif self.wasCommandIgnored(self.last_user_action['action'],self.last_system_action['action']):
                reward -= 10

            if self.last_user_action["action"]=="dont-know":
                reward -= 2

            if "system_answered" in self.last_user_action:
                reward += .7

            if self.last_system_action["action"] == "START" :
                reward -= 20

            if self.last_system_action["action"] in ["info", "info-all"] and self.current_function==None:
                reward -= 20


        return reward

    def isDone(self):
        """ Checks whether the episode is done. Considers dialogue length
            and whether user has sent the "END" act.
        """
        if self.current_turn >= self.MAX_TURNS: return True
        if self.last_user_action["action"] == "END": return True
        return False

    def getHighestScore(self):
        top_hit = self.dataset.getTopHits(1, self.result_index)
        if top_hit:
            return self.dataset.function_scores[self.dataset.function_map[top_hit[0]]]
        else:
            return 0

    def flattenObservation(self, observation):
        return flatten(self.observation_space, observation)

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        if self.current_turn == 1:
            print("Simulated user parameters")
            print("Desired func: " + str(self.user.constraints["target_function"]))
            print("Function description: ")
            print(self.dataset.functions[self.dataset.function_list[self.user.constraints["target_function"]]])
            print("Interest threshold: " + str(self.user.constraints["interest_threshold"]))
            print("Epsilon: " + str(self.user.constraints["epsilon"]))
            print("Query length: " + str(self.user.constraints["max_query_length"]))
            print()
            print("\tTurn: 0")
            print("\t\tApiza: START")
            # print("\t\t\tFunc similarity: %f"%self.dataset.results[self.user.constraints['target_function']])
            print("\t\tUser: " + self.first_response['action'])
            if "query" in self.first_response:
                print("\t\t\tQuery: %s"%self.first_response["query"])
            print()

        print("\tTurn: "+ str(self.current_turn))


        print("\t\tApiza: %s"%self.last_system_action["action"] )
        if "function" in self.last_system_action:
            print("\t\t\tFunction: %s"%self.last_system_action["function"])
        elif "list" in self.last_system_action:
            print("\t\t\tList: %s"%self.last_system_action["list"])
        print("\t\t\tFunction identified: %s"%str(self.user.constraints["target_function"] in [f[0] for f in self.user.goals["candidates"]]))
        print()

        print("\t\tUser: %s"%self.last_user_action["action"] )
        print("\t\t\tAgenda: %s"%str(self.user.goals["candidates"]))
        if "query" in self.last_user_action:
            print("\t\t\tQuery: %s"%self.last_user_action["query"])
        if "keyword" in self.last_user_action:
            print("\t\t\tKeyword: %s"%self.last_user_action["keyword"])
        print()

        print("\t\t\tCurrent function: %s"%self.current_function)
        # print("\t\t\tCommand ignored: %s"%self.command_ignored)
        print("\t\t\tDon't know: %s"%self.dont_know)
        print("\t\t\tResults: %i"%len(self.dataset.function_rankings))
        if len(self.dataset.function_rankings)>0:
            print("\t\t\tHighest score: %f"%self.dataset.function_scores[self.dataset.function_map[self.dataset.getTopHits(1)[0]]])
        print("\t\t\tFunction ranking: %i"%self.dataset.getPosition(self.user.constraints['target_function']))

        print()


        return

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    @property
    def nb_obs(self):
        return self.observation_space.shape

    @property
    def nb_actions(self):
        return self.action_space.shape

    @property 
    def last_user_action(self):
        return self.history['user_action']

    @property 
    def last_system_action(self):
        return self.history['system_action']

    @property
    def functions(self):
        return self.dataset.functions
    



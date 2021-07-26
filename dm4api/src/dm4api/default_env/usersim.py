from .dataset import Dataset

import pickle
import nltk
import random
import copy
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from re import finditer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from operator import itemgetter


'''
    TODOs:
        * Bigram probabilities for "eli-elaborate" system action are hardcoded--
        need to go through annotations and mark that where appropriate

        * "eli-sugg-related" not implemented
'''


class UserSim:

    FUNCTION_START_RATE = .25
    MAX_EPSILON = .1
    MAX_QUERY_LENGTH = 12
    MAX_LEARNING_RATE = .25
    MAX_ERROR_RATE = .2

    CORE_USER_TYPES=['info', 'eli-info', 'eli-sugg', 'elaborate', 'dont-know', 'END']
    USER_TYPES=['info', 'eli-info', 'eli-info-all','eli-sugg', 'eli-sugg-all', 'eli-sugg-related', 'elaborate', 'dont-know', 'END']

    CORE_AGENT_TYPES=['info','sugg', 'eli-info', 'eli-elaborate', "START"]
    AGENT_TYPES=['info','info-all','sugg','sugg-all','sugg-info-all','eli-info', 'eli-elaborate', "START"]  

    da_sequences = {}
    bigram_probs = {}
    support = {}

    dataset = None

    goals = None
    constraints = None
    recent_action = None

    starting_seed = random.Random()
    init_randomizer = random.Random()

    def __init__(self, dataset, sequences, annotations):
        self.dataset = dataset
        self.da_sequences = self.buildseqs(sequences, annotations, "illocutionary")
        self.bigram_probs, self.support = self.buildBigramProbs()
        self.reset()

    ##################################User DA KB####################################

    def reset_seed(self, seed):
        self.starting_seed.seed(seed)

    def reset(self, seed=None):
        a=self.starting_seed.random()
        self.init_randomizer.seed(a)
        self.goals = self.initGoals()
        self.constraints = self.initConstraints()
        self.recent_action = None

    def initGoals(self):
        """
            This method chooses the slots that the user initially wants to fill. 
            These can change every step through the "update_state" method.
        """
        goals = {}
        goals['candidates'] = []
        goals['satisfied'] = False
        return goals

    def initConstraints(self):
        """
            Choose the function that the user is looking for and generate a vector representation.
            Also initialize some other constraints, such as #turns passed, whether or not a function has been identified, 
            candidate functions that the system has dentified, etc.
        """
        constraints = {}
        # User behavior parameters
        constraints['max_query_length'] = self.init_randomizer.randrange(6,self.MAX_QUERY_LENGTH)
        constraints['epsilon'] = .00001+self.init_randomizer.random()*self.MAX_EPSILON
        constraints['learning_rate'] = 1 + (self.init_randomizer.random()*self.MAX_LEARNING_RATE)
        constraints['interest_rate'] = self.init_randomizer.random()*.2
        constraints['keyword_error_rate'] = self.init_randomizer.random()*self.MAX_ERROR_RATE
        constraints['interest_threshold'] = self.init_randomizer.random()*.8 + .2
        constraints['info_threshold'] = self.init_randomizer.random()*.3
        constraints['elaborate_threshold'] = self.init_randomizer.random()*.6 + .6
        constraints['eli_sugg_all_threshold'] = self.init_randomizer.random() 
        # User knowledge state
        constraints['list_current'] = False
        constraints['provided_query'] = False
        constraints['additional_information'] = self.init_randomizer.random()+self.init_randomizer.random()
        constraints['can_info'] = constraints['additional_information']>=constraints['info_threshold']
        constraints['can_elaborate'] = constraints['additional_information']>=constraints['elaborate_threshold']
        #Desired function information
        constraints['target_function'] = self.dataset.getRandomFunction(random_object=self.init_randomizer)
        constraints['target_function_named'] = self.dataset.function_list[constraints['target_function']]
        constraints['target_query'] = ""
        constraints['incorrect_keywords'] = set()

        return constraints


    def buildseqs(self, sequences, annotations, feature, lib=None):
        if lib: dialogue_list = [key for key in list(annotations) if annotations[key]["lib"] == lib]
        else: dialogue_list = list(annotations)
        da_sequences = {}
        for key in dialogue_list:
            da_sequences[key] = self.buildSeqsDia(sequences[key], annotations[key]["utterances"], feature)
        return da_sequences

    def buildSeqsDia(self, sequences, annotations, feature):
        da_sequences = []
        for thread in sequences:
            da_thread = []
            for utterances in thread:
                das = [(annotations[rid]['speaker'], annotations[rid][feature])for rid in utterances]
                da_thread.append(das)
            da_sequences.append(da_thread)
        return da_sequences

    def reduce(self, da):
        for core_type in self.CORE_USER_TYPES:
            if da[1].startswith(core_type):
                return (da[0], core_type)
        return (da[0], "OTHER")

    def reduceAgent(self, da):
        for core_type in self.CORE_AGENT_TYPES:
            if da.startswith(core_type):
                return core_type
        return "OTHER"

    def buildBigramProbs(self):
        bigram_counts = {}
        for da_sequences_dia in self.da_sequences.values():
            for da_sequence in da_sequences_dia:
                da_sequence.insert(0, [("Apiza", "START")])
                da_sequence.append([("Programmer", "END")])
                da_sequence_len = len(da_sequence)
                for i in range(da_sequence_len-1):
                    apiza_checked = set()
                    for da in da_sequence[i]:
                        speaker, act = da
                        if speaker == "Apiza" and act not in apiza_checked:
                            apiza_checked.add(act)
                            if act not in bigram_counts: bigram_counts[act] = Counter()
                            user_checked = set()
                            for next_da in da_sequence[i+1]:
                                next_speaker, next_act = self.reduce(next_da)
                                if next_speaker == "Programmer" and next_act not in user_checked:
                                    user_checked.add(next_act)
                                    bigram_counts[act][next_act] += 1
        bigram_counts["eli-elaborate"] = Counter()
        bigram_counts["eli-elaborate"]["elaborate"] = 4
        bigram_counts["eli-info"]["info"] *= 5

        for da in bigram_counts:
            for response in self.CORE_USER_TYPES:
                bigram_counts[da][response] += .3

        for da in bigram_counts:
            total_responses = sum(bigram_counts[da].values())
            for response in bigram_counts[da]:
                bigram_counts[da][response] = bigram_counts[da][response]/total_responses

        support = copy.deepcopy(bigram_counts)

        return bigram_counts, support

    ##################################User DA Decisions####################################

    def respond(self, system_action):
        ''' 
            Selects a dialogue act type and collects any updates query text and vector
            
            Uses the learned DA bigram probabilities to select a basic DA type, 
            then uses some heuristics to select a specific variant if needed
        '''

        self.processSystemAction(system_action)
        self.processCandidates()
        self.processSharing()

        user_action = {}
        if self.goals['satisfied']:
            user_action = {"action":"END"}
        else:
            user_action = self.createUserAction(system_action)
        return user_action

    def processSystemAction(self, system_action):
        """ 
            The purpose of this function is to update the user's internal state
            in response to the system action. 
        """
        def applyInterestUpdate(function, interest_update):
            for i in range(len(self.goals["candidates"])):
                if self.goals["candidates"][i][0] == function:
                    self.goals["candidates"][i][1] += interest_update
                    return

        def info():
            function = system_action['function']
            interest_update = 0
            # Calculate interest update
            if function == self.constraints['target_function_named']:
                interest_update = self.init_randomizer.random()
            else:
                interest_update = -self.init_randomizer.random()
            applyInterestUpdate(function,interest_update)
            # Increase additional information meter slightly
            self.constraints['additional_information'] += self.init_randomizer.random()*.75

        def infoAll():
            function = system_action['function']
            interest_update = 0
            # Calculate interest update
            if function == self.constraints['target_function_named']:
                interest_update = self.init_randomizer.random()*2
            else:
                interest_update = -self.init_randomizer.random()*2
            applyInterestUpdate(function,interest_update)
            # Increase additional information meter a good bit
            self.constraints['additional_information'] += self.init_randomizer.random()

        def sugg():
            function = system_action['function']
            # Add it to the candidate queue if correct or lucky
            if function == self.constraints['target_function_named']:
                interest_level = max(self.init_randomizer.random(), self.constraints["interest_threshold"])
            else:
                interest_level = self.init_randomizer.random()
            if interest_level >= self.constraints["interest_threshold"] and (function == self.constraints['target_function_named'] or len(self.goals["candidates"])<10) and function not in {f[0] for f in self.goals["candidates"]}:
                self.goals["candidates"].append([function, interest_level])
            # Increase additional information meter slightly
            self.constraints['additional_information'] += self.init_randomizer.random()*.25


        def suggAll():
            for function in system_action['list']:
                # Add it to the candidate queue if correct or lucky
                if function == self.constraints['target_function_named']:
                    interest_level = max(self.init_randomizer.random(), self.constraints["interest_threshold"])
                else:
                    interest_level = self.init_randomizer.random()
                if interest_level >= self.constraints["interest_threshold"] and (function == self.constraints['target_function_named'] or len(self.goals["candidates"])<10) and function not in {f[0] for f in self.goals["candidates"]}:
                    self.goals["candidates"].append([function, interest_level])
            self.constraints['additional_information'] += self.init_randomizer.random()*.5
            self.constraints["list_current"] = True

        def suggInfo():
            # Mimics behavior of Apiza suggesting a new function, and giving all infromation about it
            sugg()
            infoAll()

        def eliKw():
            # Mimics behavior of Apiza eliciting information
            # Doesn't actually update the state, beyond letting the user know Apiza wants information
            return

        def eliQuery():
            # Mimics behavior of Apiza eliciting information
            # Doesn't actually update the state, beyond letting the user know Apiza wants information
            return

        def other():
            # Does nothing, how did this act even get chosen?
            return
        
        switcher = {
            'info': info,
            'info-all': infoAll,
            'sugg': sugg,
            'sugg-all': suggAll,
            'sugg-info-all': suggInfo,
            'change-page': suggAll,
            'eli-kw': eliKw,
            'eli-query': eliQuery,
            'OTHER': other,
            "START": other
        }
        switcher[system_action['action']]()
        return

    def processCandidates(self):
        # Improve query if solved candidate by inform acts
        len_can = len(self.goals["candidates"])
        self.goals["candidates"] = [c for c in self.goals["candidates"] if c[1]>0]
        len_lost = len_can - len(self.goals["candidates"])
        if len_lost:
            self.constraints["epsilon"] = self.constraints["epsilon"]/(self.constraints["learning_rate"]**len_lost)
        
        #Lose interest over time by default
        for i in range(len(self.goals["candidates"])):
            if self.goals["candidates"][i][0] != self.constraints["target_function_named"]:
                self.goals["candidates"][i][1] -= self.init_randomizer.random()*self.constraints["interest_rate"]
            else:
                self.goals["candidates"][i][1] += self.init_randomizer.random()*self.constraints["interest_rate"]*(1-self.goals["candidates"][i][1])
        self.goals["candidates"] = [c for c in self.goals["candidates"] if c[1]>0]
        
        if self.goals["candidates"]:
            self.goals["candidates"].sort(key=itemgetter(1), reverse=True)
            if self.goals["candidates"][0][0] == self.constraints["target_function_named"] and self.goals["candidates"][0][1] >= 1:
                self.goals["satisfied"] = True

    def processSharing(self):
        if self.constraints["additional_information"] < 0:
            self.constraints["can_elaborate"] = False
            self.constraints["can_info"] = False
        else:
            if self.constraints["additional_information"] > self.constraints["elaborate_threshold"]:
                self.constraints["can_elaborate"] = True
            if self.constraints["additional_information"] > self.constraints["info_threshold"]:
                self.constraints["can_info"] = True

    def createUserAction(self, system_action):
        user_action = {}

        if system_action["action"] == "START":
            query_length = 1+self.init_randomizer.randrange(self.constraints['max_query_length'])
            new_query =  self.dataset.getQuery(self.constraints['target_function'], query_length, self.constraints['epsilon'], random_object=self.init_randomizer)
            if self.init_randomizer.random()>self.FUNCTION_START_RATE:
                self.constraints["additional_information"] -= 2*self.init_randomizer.random()
                self.constraints["list_current"] = False
                self.constraints["target_query"] = new_query
                user_action["query"] = self.constraints['target_query']
                user_action["action"] = "provide-query"
                # print("UA1")
                return user_action
            else:
                self.constraints['additional_information']=max(self.constraints['additional_information'], self.constraints['elaborate_threshold'])
                first_candidate = self.dataset.getSimilarFunction(new_query, random_object=self.init_randomizer)
                self.goals["candidates"].append([first_candidate, self.init_randomizer.random()])
                # print("UA2")

        # Encountered an empty results page, try to remove some incorrect keywords
        if (((system_action["action"] in ["sugg-all", "change-page"] and not system_action["list"]) or 
                                (system_action["action"] in ["sugg", "sugg-info-all"] and not system_action["function"])) 
                                and self.constraints["incorrect_keywords"]):
            keyword = self.init_randomizer.sample(self.constraints["incorrect_keywords"], 1)[0]
            self.constraints["incorrect_keywords"].remove(keyword)
            user_action["keywords"] = [keyword]
            user_action["action"] = "reject-kws"
            # print("UA3")
        else:
            actions = self.chooseWeightedResponse(system_action["action"])
            while "action" not in user_action:
                if actions[0] == "info":
                    if system_action["action"] == "eli-kw":
                        if self.constraints['additional_information']>=self.constraints['info_threshold']:
                            # Provide incorrect keyword, or incorrectly reject keywords
                            if self.init_randomizer.random() < self.constraints["keyword_error_rate"]/2:
                                worst_keyword = self.dataset.getWorstKeyword(self.constraints["target_function"], system_action["keywords"], self.init_randomizer)
                                if worst_keyword:
                                    self.constraints["list_current"] = False
                                    self.constraints["incorrect_keywords"].add(worst_keyword)
                                    user_action["keyword"] = worst_keyword
                                    user_action["action"] = "provide-kw"
                                    # print("UA04")
                                    user_action["system_answered"] = True
                                else:
                                    user_action["keywords"] = system_action["keywords"]
                                    user_action["action"] = "reject-kws"
                                    # print("UA05")
                            #Provide correct kw, or correctly reject them
                            else:
                                self.constraints["additional_information"] -= self.init_randomizer.random()*.3
                                best_keyword = self.dataset.getBestKeyword(self.constraints["target_function"], system_action["keywords"])
                                if best_keyword:
                                    self.constraints["list_current"] = False
                                    user_action["keyword"] = best_keyword
                                    user_action["action"] = "provide-kw"
                                    # print("UA06")
                                    user_action["system_answered"] = True
                                else:
                                    user_action["keywords"] = system_action["keywords"]
                                    user_action["action"] = "reject-kws"
                                    # print("UA07")
                        # Trying to answer kw question but can't
                        else:
                            user_action["action"] = "dont-know"
                            user_action["query"] = self.constraints['target_query']
                            # print("UA08")
                    elif system_action["action"] in ["sugg-all", "change-page"] and self.constraints["target_function_named"] not in system_action["list"]:
                        user_action["action"] = "reject-functions"
                        user_action["functions"] = system_action["list"]
                        # print("UA09")
                    else:
                        # Unprompted, fixing incorrect keywords
                        if self.constraints["incorrect_keywords"]:
                            keyword = self.init_randomizer.sample(self.constraints["incorrect_keywords"], 1)[0]
                            self.constraints["incorrect_keywords"].remove(keyword)
                            user_action["keywords"] = [keyword]
                            user_action["action"] = "reject-kws"
                            # print("UA10")
                        #Unprompted, providing keywords
                        else:
                            if self.init_randomizer.random() < self.constraints["keyword_error_rate"]:
                                keyword = self.dataset.getKeyword(self.constraints["target_function"], wrong=True, random_object=self.init_randomizer)
                                if keyword:
                                    self.constraints["incorrect_keywords"].add(keyword)
                                # print("UA11A")
                            else:
                                keyword = self.dataset.getKeyword(self.constraints["target_function"], random_object=self.init_randomizer)
                                # print("UA11B")
                            if keyword:
                                # print("UA11")
                                self.constraints["list_current"] = False
                                user_action["keyword"] = keyword
                                user_action["action"] = "provide-kw"
                    actions.pop(0)


                elif actions[0] == "elaborate":
                    #Provide query, if we can
                    if self.constraints['additional_information']>=self.constraints['elaborate_threshold']:
                        query_length = 1+self.init_randomizer.randrange(self.constraints['max_query_length'])
                        new_query =  self.dataset.getQuery(self.constraints['target_function'], query_length, self.constraints['epsilon'], random_object=self.init_randomizer)
                        self.constraints["additional_information"] -= 2*self.init_randomizer.random()
                        self.constraints["list_current"] = False
                        self.constraints["target_query"] = new_query
                        user_action["query"] = self.constraints['target_query']
                        user_action["action"] = "provide-query"
                        # print("UA12")
                    else:
                        user_action["action"] = "dont-know"
                        # print("UA13")


                elif actions[0] == "eli-info":
                    #Elicit info, if there's an interested function
                    if self.goals["candidates"]:
                        if self.init_randomizer.random()<self.goals["candidates"][0][1]:
                            user_action["action"] = "eli-info"
                            # print("UA14")
                        else:
                            user_action["action"] = "eli-info-all"
                            # print("UA15")
                        user_action["function"] = self.goals["candidates"][0][0]
                    else:
                        actions.pop(0)

                elif actions[0] == "eli-sugg":
                    # Elicit suggestions, any time.
                    if self.constraints['additional_information']<self.constraints['info_threshold'] or self.init_randomizer.random()<self.constraints["eli_sugg_all_threshold"]:
                        if self.constraints["list_current"]:
                            user_action["action"] = "change-page"
                            # print("UA16")
                        else:
                            user_action["action"] = "eli-sugg-all"
                            # print("UA17")
                    else: 
                        user_action["action"] = "eli-sugg"
                        # print("UA18")
                else:
                    actions.pop(0)
        return user_action
    




    def weighted_shuffle(self, items, weights, epsilon=0, random_object=None):
        if random_object:
            order = sorted(range(len(items)), key=lambda i: random_object.random() ** (1.0 / (weights[i]+epsilon)), reverse=True)
        else:
            order = sorted(range(len(items)), key=lambda i: random.random() ** (1.0 / (weights[i]+epsilon)), reverse=True)
        return [items[i] for i in order]

    def chooseWeightedResponse(self, da):
        if da == "eli-query":
            da = "eli-elaborate"
        elif da == "eli-kw":
            da = "eli-info"
        elif da == "change-page":
            da = "sugg-all"
        if da not in self.bigram_probs: 
            da = self.reduceAgent(da)
            if da not in self.bigram_probs: 
                print("mysterious da? %s"%da)
                return self.chooseRandomResponse()
        items, weights = zip(*self.bigram_probs[da].items())
        return self.weighted_shuffle(items,weights, random_object=self.init_randomizer)

    def chooseOptimalResponse(self, da):
        options = self.bigram_probs[da]
        response = max(options, key=options.get) 
        return response

    def chooseRandomResponse(self):
        core_user_types = self.CORE_USER_TYPES.copy()
        self.init_randomizer.shuffle(core_user_types)
        return core_user_types

    #####################################Testing##########################################

    def testBigramModel(self):
        da_sequences=self.da_sequences
        bigram_probs=self.bigram_probs
        num_guesses = 0
        num_correct_weighted = 0
        num_correct_optimal = 0
        for da_sequences_dia in da_sequences.values():
            for da_sequence in da_sequences_dia:
                da_sequence.insert(0, [("START", "Apiza")])
                da_sequence.append([("END", "Programmer")])
                da_sequence_len = len(da_sequence)
                for i in range(da_sequence_len-1):
                    for da in da_sequence[i]:
                        for next_da in da_sequence[i+1]:
                            if da[0] == "Apiza" and next_da[0] == "Programmer":
                                true_response = next_da[1]
                                weighted_response = self.chooseWeightedResponse(da[1])
                                optimal_response = self.chooseOptimalResponse(da[1])
                                if true_response == weighted_response:
                                    num_correct_weighted += 1
                                if true_response == optimal_response:
                                    num_correct_optimal += 1
                                num_guesses += 1
        print(num_correct_weighted/num_guesses)
        print(num_correct_optimal/num_guesses)

    def printSupport(self):
        for k in self.support:
            print("{}: {}".format(k, self.support[k]))
            print()

    def printBigramProbs(self):
        for k in self.bigram_probs:
            print("{}: {}".format(k, self.bigram_probs[k]))
            print()


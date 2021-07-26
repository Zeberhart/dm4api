import pickle
import nltk
import random
import copy
import math
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from re import finditer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

class Dataset:

    K = 2
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))

    function_list = None
    vocab = None
    function_keys = None
    vectors=None
    vectorizer=None
    feature_names = None

    stem_dict = defaultdict(set)

    function_search_vectors = None

    function_scores = None
    function_rankings = None

    query_scores = None
    not_function_scores = None
    keyword_scores = None


    def __init__(self, api):
        self.functions = api["components"]
        self.function_list  = list(self.functions)
        self.function_map  = {function: i for i, function in enumerate(self.function_list)}
        self.num_functions  = len(self.function_list)

        self.tfidf_vectorizer = self.buildTfidfVectorizer()
        self.feature_names = self.tfidf_vectorizer.get_feature_names()
        self.functions_tfidf = self.buildTfidfDict()
        self.function_search_vectors = self.buildFunctionSearchMatrix() 

        self.reset()

    def buildTfidfVectorizer(self):
        # Build the TFIDF vectorizer, considering both categories and functions when calculating IDF
        names = []
        corpus = []
        for name, facets in self.functions.items(): 
            names.append(name)
            text = ""
            for facet in facets:
                if isinstance(facets[facet], str):
                    text += " " + facets[facet]
                elif isinstance(facets[facet], list):
                    text += " " + " ".join(facets[facet])
            text = self.process_text(text, building_corpus=True)
            corpus.append(text)
        vectorizer = TfidfVectorizer(max_df=.5)
        vectorizer.fit_transform(corpus)
        return vectorizer

    def buildTfidfDict(self):
        tfidf_itemset = {}
        for name, facets in self.functions.items():
            tfidf_itemset[name] = {}
            for facet in facets:
                if isinstance(facets[facet], str):
                    tfidf_itemset[name][facet] = self.tfidf_vectorizer.transform([self.process_text(facets[facet])])[0].toarray()[0]
                elif isinstance(facets[facet], list):
                    tfidf_itemset[name][facet] = self.tfidf_vectorizer.transform([self.process_text(" ".join(facets[facet]))])[0].toarray()[0]
        return tfidf_itemset

    def buildFunctionSearchMatrix(self):
        matrix = []
        for name in self.function_list:
            signature_vec = self.functions_tfidf[name]["Signature"]
            description_vec = self.functions_tfidf[name]["Description"]
            full_vec = self.functions_tfidf[name]["All"]
            search_vec = np.average([signature_vec,description_vec,full_vec], axis=0)
            matrix.append(search_vec)
        return np.array(matrix)


    def reset(self):
        self.query = ""
        self.keywords = {"provided":set(), "rejected":set()}
        self.not_functions = set() 

        self.query_scores= np.ones(self.getDatabaseSize())
        self.not_function_scores = np.ones(self.getDatabaseSize())
        self.keyword_scores = np.ones(self.getDatabaseSize())

        self.function_scores = np.ones(self.getDatabaseSize())
        self.function_rankings = []

    ##################################Function Knowledge Base###############################

    def process_text(self, text, building_corpus=False):
        text=text.replace("_", " ")
        tokens = [st.lower() for t in self.tokenizer.tokenize(text) for st in self.camel_and_snake_case_split(t) if st.strip()]
        tokens = [t for t in tokens if t not in self.stop_words]
        if building_corpus:
            for t in tokens:
                self.stem_dict[self.ps.stem(t)].add(t)
        tokens = [self.ps.stem(t) for t in tokens]
        return " ".join(tokens)

    def process_tokens(self, tokens):
        tokens = [self.ps.stem(t) for t in tokens]
        return tokens

    def camel_and_snake_case_split(self, identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def getMaxDistance(self):
        return self.function_search_vectors.shape[1]**.5

    def getDatabaseSize(self):
        return self.function_search_vectors.shape[0]

    def getVocabularySize(self):
        return len(self.tfidf_vectorizer.vocabulary_)

    def getRandomFunction(self, random_object=None):
        if random_object:
            return random_object.randint(0, self.getDatabaseSize()-1)
        else:
            return random.randint(0, self.getDatabaseSize()-1)

    def getSimilarFunction(self, query, random_object=None):
        query_vector = csr_matrix([self.getVectorForQuery(query)])
        query_scores = cosine_similarity(query_vector, self.function_search_vectors, dense_output=False)[0]
        func_index = self.weighted_shuffle(range(len(query_scores)), query_scores, epsilon=0.001, random_object=random_object)[0]
        return self.function_list[func_index]

    def getVector(self, function):
        return self.function_search_vectors[function]

    # def getMutatedVector(self, function, probability=.3, magnitude=.5):
    #     vector = self.getVector(function)
    #     print(vector)
    #     indices = vector.nonzero()[1]
    #     vector = vector.toarray()[0]
    #     for index in range(len(vector)):
    #         if random.random()<probability:
    #             if index in indices:
    #                 vector[index] -= random.random()*magnitude
    #             else: 
    #                 vector[index] += random.random()*magnitude 
    #     return csr_matrix([vector])

    def weighted_shuffle(self, items, weights, epsilon=.05, random_object=None):
        if random_object:
            order = sorted(range(len(items)), key=lambda i: random_object.random() ** (1.0 / (weights[i]+epsilon)), reverse=True)
        else:
            order = sorted(range(len(items)), key=lambda i: random.random() ** (1.0 / (weights[i]+epsilon)), reverse=True)
        return [items[i] for i in order]

    def decodeVocab(self, tokens):
        return [random.choice(tuple(self.stem_dict[self.feature_names[t]])) for t in tokens]

    def getQuery(self, function, length=5, epsilon=.01, random_object=None):
        function_vector = self.function_search_vectors[function]
        sorted_tokens = self.weighted_shuffle(list(range(len(function_vector))), function_vector, epsilon, random_object=random_object)
        query = " ".join(self.decodeVocab(sorted_tokens[:length]))
        return query

    def getVectorForQuery(self, query, process=True):
        if process:
            query = self.process_text(query)
        vector = self.tfidf_vectorizer.transform([query])
        return vector[0].toarray()[0]

    # def getVectorForQuery(self, query):
    #     query = self.process_text(query)
    #     vector = np.zeros(self.getVocabularySize())
    #     for token in query.split():
    #         if token in self.tfidf_vectorizer.vocabulary_:
    #             vector[self.tfidf_vectorizer.vocabulary_[token]] = 1
    #     return vector

    ##################################Function Searching###############################

    def updateResults(self, query=None, keywords=None, not_functions=None):

        if query:
            self.query = query
            self.expanded_query = query+ " " + " ".join(self.keywords["provided"])
            self.query_vector = csr_matrix([self.getVectorForQuery(self.expanded_query)])
            self.query_scores = cosine_similarity(self.query_vector, self.function_search_vectors, dense_output=False)[0]

        elif keywords:
            self.keywords = keywords
            if(keywords["provided"]):
                keywords_provided_vector = self.getVectorForQuery(" ".join(keywords["provided"]))
                keywords_provided_indices = keywords_provided_vector.nonzero()[0]
                if len(keywords_provided_indices)>0:
                    self.keyword_scores = [int(np.amin(function_vector[keywords_provided_indices])>0) for function_vector in self.function_search_vectors]
                else:
                    self.keyword_scores = np.ones(self.getDatabaseSize())
            else:
                self.keyword_scores = np.ones(self.getDatabaseSize())

            keywords_string = " ".join(keywords["provided"])
            keywords_processed = set(self.process_text(keywords_string).split())
            keywords_sring_processed = " ".join(keywords_processed)
            query_processed = self.process_text(self.query)
            self.expanded_query = query_processed+ " " + keywords_sring_processed
            self.query_vector = csr_matrix([self.getVectorForQuery(self.expanded_query, process=False)])
            self.query_scores = cosine_similarity(self.query_vector, self.function_search_vectors, dense_output=False)[0]

        elif not_functions:
            self.not_functions = not_functions
            not_function_indices = [self.function_map[function] for function in not_functions]
            self.not_function_scores = [0 if i in not_function_indices else 1 for i in range(self.num_functions)]

        self.function_scores = np.prod(np.array([self.query_scores, self.keyword_scores, self.not_function_scores]), axis=0)
        l = zip(self.function_scores.nonzero()[0], self.function_scores[self.function_scores.nonzero()[0]])
        top = sorted(l, reverse=True, key=lambda v: v[1])
        self.function_rankings = [self.function_list[key] for key,val in top]

    # def fixPage(self, K, page):
    #     if K*(page-1)> len(self.function_rankings):
    #         return math.ceil(len(self.function_rankings)/K)
    #     elif page < 1:
    #         return 1
    #     else:
    #         return page

    def getTopHits(self, K=10, result_index=0):
        if result_index<len(self.function_rankings):
            return self.function_rankings[result_index: result_index+K]
        else:
            return []

    def getSuggestedKeywords(self):
        function_rankings = self.function_rankings
        if not function_rankings:
            function_rankings = self.function_list

        top_function_vectors = self.function_search_vectors[[self.function_map[function] for function in function_rankings[:20]]]
        average_top_function_vector = np.average(top_function_vectors, axis=0)

        if self.query:
            query_indices = self.query_vector.nonzero()[1]
            average_top_function_vector[query_indices] = 0
        if self.keywords:
            keyword_vector = self.getVectorForQuery(" ".join(list(self.keywords["provided"])+list(self.keywords["rejected"])))
            keyword_indices = keyword_vector.nonzero()[0]
            average_top_function_vector[keyword_indices] = 0

        l = zip(average_top_function_vector.nonzero()[0], average_top_function_vector[average_top_function_vector.nonzero()[0]])
        top = sorted(l, reverse=True, key=lambda v: v[1])
        return self.decodeVocab([key for key,val in top])

    def checkRelevantKeywords(self, function, keywords):
        #Function is int, keywords are decoded
        relevant_keywords = []
        keywords = list(keywords)
        keyword_string = " ".join(keywords)
        processed_keyword_string = self.process_text(keyword_string)
        for i, token in enumerate(processed_keyword_string.split()):
            if token in self.tfidf_vectorizer.vocabulary_:
                index = self.tfidf_vectorizer.vocabulary_[token]
                if self.function_search_vectors[function][index]>0:
                    relevant_keywords.append(keywords[i])
        return relevant_keywords

    def getBestKeyword(self, function, keywords):
        #Function is int, keywords are decoded
        keywords = list(keywords)
        keyword_string = " ".join(keywords)
        processed_keyword_string = self.process_text(keyword_string)
        best_keyword_score=0
        best_keyword = None
        for i, token in enumerate(processed_keyword_string.split()):
            if token in self.tfidf_vectorizer.vocabulary_:
                index = self.tfidf_vectorizer.vocabulary_[token]
                if self.function_search_vectors[function][index]>best_keyword_score:
                    best_keyword_score = self.function_search_vectors[function][index]
                    best_keyword = keywords[i]
        return best_keyword

    def getWorstKeyword(self, function, keywords, random_object=None):
        #Function is int, keywords are decoded
        keywords = list(keywords)
        keyword_string = " ".join(keywords)
        processed_keywords= self.process_text(keyword_string).split()
        keyword_indices = [self.tfidf_vectorizer.vocabulary_[t] for t in processed_keywords if t in self.tfidf_vectorizer.vocabulary_]
        keyword_scores = [self.function_search_vectors[function][i] for i in keyword_indices]
        valid_indices = np.nonzero(np.array(keyword_indices)>=0)[0]
        if len(valid_indices)>0:
            if random_object:
                return keywords[random_object.choice(valid_indices)]
            else:
                return keywords[random.choice(valid_indices)]
        else:
            return None

    def getKeyword(self, function, wrong=False, random_object=None):
        if not wrong:
            if random_object:
                random_keyword_index = random_object.choice(self.function_search_vectors[function].nonzero()[0])
            else:
                random_keyword_index = random.choice(self.function_search_vectors[function].nonzero()[0])
        else:
            if random_object:
                random_keyword_index = random_object.choice(np.nonzero(self.function_search_vectors[function]==0)[0])
            else:
                random_keyword_index = random.choice(np.nonzero(self.function_search_vectors[function]==0)[0])
        return self.decodeVocab([random_keyword_index])[0]



    def getHighestScore(self):
        return max(self.function_scores)

    def getPosition(self, function):
        if self.function_list[function] not in self.function_rankings: return self.num_functions
        else: return self.function_rankings.index(self.function_list[function])


def main():
    functions = pickle.load(open("../../../data/new_scheme/functions.pkl", 'rb'))
    dataset = Dataset(functions)

if __name__ == "__main__":
    main()
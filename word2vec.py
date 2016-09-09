from __future__ import division
from pymongo import MongoClient
from gensim.models import Word2Vec
import numpy as np
import os,sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from collections import defaultdict
import cPickle as pickle
#import matplotlib.pyplot as plt
import string
from DeepMining.smart_search import SmartSearch
from nltk.corpus import wordnet as wn
import itertools
import feature_extractor
from pymongo import MongoClient
client = MongoClient()
metadb = client.metadb

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cdir,'..'))

class Clusterer(object):

        #TODO: augment cluster_dict with related nouns as soon as we get clusters
    def __init__(self, dataset_collection='all_datasets', word_embeddings = 'google_news',
            restrictToTimeBased = False, restrictToColumns = False,load_from_pickle=True, cluster_pickle_file = None,
            nclusters = 14, batch_size = 400, max_iter= 300, n_init=3, init_size=2000):
        if dataset_collection == 'all_datasets':
            self.datasets = metadb.all_datasets
        elif dataset_collection == 'active_datasets':
            self.datasets = metadb.active_datasets
        else:
            raise Exception("unknown dataset collection: {}".format(dataset_collection))
        if word_embeddings == 'google_news':
            self.word_embedding_dataset = os.path.join(cdir,'GoogleNews-vectors-negative300.bin')
        else:
            raise Exception("unknown word embedding dataset: {}".format(word_embedding_dataset))

        #self.model = self.getWordEmbeddingsDict()
        self.model = None

        self.restrictToTimeBased = restrictToTimeBased
        self.restrictToColumns = restrictToColumns

        if cluster_pickle_file:
            self.cluster_pickle_file = cluster_pickle_file
        else:
            self.cluster_pickle_file = "clusters%s_batchSize%s_maxIter%s_nInit%s_initSize%s.p"%(nclusters, batch_size, max_iter, n_init, init_size)


        self.nclusters = nclusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_size = init_size

        if load_from_pickle:
            if os.path.isfile(self.cluster_pickle_file):
                pobject = pickle.load(open(self.cluster_pickle_file,'rb'))
                self.cluster_dict = pobject['cluster_dict']
                self.clusters = pobject['clusters']
            else:
                self.cluster_dict = None
                self.clusters = None


    def getWordEmbeddingsDict(self):
        model = Word2Vec.load_word2vec_format(self.word_embedding_dataset, binary=True)
        return model
    def toASCII(self, w):
        try:
            return str(w)
        except:
            return None

    def separateByCapitalization(self, word):
        #possibilities:
        #1. first letter capitalized:
        #   Athlete
        #2. two or more letters capitalized surround by lower case letters or start/end of word:
        #   aeroFS (split on first of capitalized, and then first of lowercase afterward if not at end)
        #3. 1 letter capitalized within word:
        #   byProduct (split on first of capitalized)
        splits = []

        prevLetterCap = False
        prevTwoLettersCap = False
        for i,char in enumerate(word):
            if i == 0:
                splits.append(i)
            elif not char.islower() and not prevLetterCap:
                #start of a capital block
                splits.append(i)
            elif not char.islower():
                #continuation of a capital block
                prevTwoLettersCap = True
            elif prevLetterCap:
                #lowercase with prev letter cap
                if prevTwoLettersCap:
                    splits.append(i)
                prevLetterCap = False
                prevTwoLettersCap = False
            else:
                #lower case without prev letter cap
                #just setting these for clarity
                prevLetterCap = False
                prevTwoLettersCap = False

            if not char.islower():
                prevLetterCap = True
        words = []
        for i, split in enumerate(splits[:-1]):
            words.append(word[split:splits[i+1]])
        #get the last split
        words.append(word[splits[-1]:])
        return words

    def _format(self, w):
        w = self.toASCII(w)
        if not w:
            return None
        else:
            translation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
            _words = string.translate(w,translation).split(' ')
            words = []
            for w in _words:
                words.extend(w.split('_'))
            _words = words
            words = []
            for w in _words:
                words.extend(w.split('-'))
            words = [w for w in words if len(w) > 1]
            words = [w.lower() for word in words for w in self.separateByCapitalization(word)]
            return words

    def getWordVectorsForDatasets(self, restrictToTimeBased = None, restrictToColumns = None):
        if restrictToTimeBased == None:
            restrictToTimeBased = self.restrictToTimeBased
        if restrictToColumns == None:
            restrictToColumns = self.restrictToColumns

        if not self.model:
            self.model = self.getWordEmbeddingsDict()
        datasets = self.datasets.find()
        dataset_words = {}
        dataset_embeddings = {}
        for dataset in datasets:
            if 'garbage' in dataset and dataset['garbage']:
                continue
            if restrictToTimeBased and not self.checkIfTimeBased(dataset):
                continue
            _id = dataset['_id']
            dataset_embeddings[_id] = {}
            dataset_words = self.getWordsFromDataset(dataset, restrictToColumns)
            for word in dataset_words:
                if word in self.model:
                    dataset_embeddings[_id][word] = self.model[word]
                # else:
                    # print word
        return dataset_embeddings

    def checkIfTimeBased(self, dataset):
        for f in dataset['files']:
            table = dataset['files'][f]
            for c in table['columns']:
                dtype = table['columns'][c]['col_type']
                if dtype == 'date':
                    return True
        return False
    def getWordsFromDatasets(self, restrictToTimeBased = None, restrictToColumns = None):
        if restrictToTimeBased == None:
            restrictToTimeBased = self.restrictToTimeBased
        if restrictToColumns == None:
            restrictToColumns = self.restrictToColumns

        datasets = self.datasets.find()
        dataset_words = {}
        for dataset in datasets:
            if 'garbage' in dataset and dataset['garbage']:
                continue
            if restrictToTimeBased and not self.checkIfTimeBased(dataset):
                continue
            _id = dataset['_id']
            dataset_words[_id] = self.getWordsFromDataset(dataset, restrictToColumns)
        return dataset_words

    def separateWord(self, word):
        for w in word.split(' '):
            yield w
    def getWordsFromDataset(self, dataset, restrictToColumns):
        dataset_words = set()
        if not restrictToColumns and 'name' in dataset:
            words = self._format(dataset['name'])
            if words:
                for word in words:
                    dataset_words.add(word)
        if not restrictToColumns and 'description' in dataset:
            words = self._format(dataset['description'])
            if words:
                for word in words:
                    dataset_words.add(word)
        for f in dataset['files']:
            table = dataset['files'][f]
            if not restrictToColumns:
                if 'tablename' in dataset['files'][f]:
                    tablename = table['tablename']
                    words = self._format(table['tablename'])
                    if words:
                        for word in words:
                            dataset_words.add(word)
            for c in table['columns']:
                column = table['columns'][c]
                words = self._format(column['col_name'])
                if words:
                    for word in words:
                        dataset_words.add(word)
        return dataset_words

    def clusterDatasets(self, from_pickle=True, restrictToColumns = None, restrictToTimeBased = None, nclusters=None,
            batch_size=None, max_iter=None, n_init=None, init_size=None, cluster_pickle_file=None):
        if self.cluster_dict:
            return self.cluster_dict, self.clusters
        if restrictToTimeBased == None:
            restrictToTimeBased = self.restrictToTimeBased
        if restrictToColumns == None:
            restrictToColumns = self.restrictToColumns
        dataset_embeddings = self.getWordVectorsForDatasets(restrictToTimeBased = restrictToTimeBased, restrictToColumns = restrictToColumns)
        X = np.array([i.T for _id in dataset_embeddings for i in dataset_embeddings[_id].itervalues()])
        y = [i for _id in dataset_embeddings for i in dataset_embeddings[_id].iterkeys()]

        # parameters = {"n_clusters": ['int',[3, 100]],
                        # "batch_size": ['int',[50,500]],
                        # "max_iter": ['int',[299,300]],
                        # "n_init": ['int',[3,4]],
                        # "init_size": ['int',[500,3000]]}
        # def scoring_function(parameters):
            # clusters = MiniBatchKMeans(**parameters)
            # clusters.fit(X)
            # return [-1*clusters.inertia_ - (parameters["n_clusters"])*(clusters.inertia_/500.)]
        # search = SmartSearch(parameters,estimator=scoring_function,n_iter=30)
        # all_parameters,all_raw_outputs = search._fit()
        # print all_parameters
        # print all_raw_outputs


        if not nclusters:
            nclusters = self.nclusters
        if not batch_size:
            batch_size = self.batch_size
        if not max_iter:
            max_iter = self.max_iter
        if not n_init:
            n_init = self.n_init
        if not init_size:
            init_size = self.init_size
        clusters = MiniBatchKMeans(n_clusters=nclusters, max_iter = max_iter,
                                        batch_size=batch_size,n_init=n_init,init_size=init_size)
        clusters.fit(X)
        self.cluster_dict={}
        self.clusters = defaultdict(set)
        for word,label in zip(y,clusters.labels_):
            self.cluster_dict[word] = label
            self.clusters[label].add(word)
        self.augmentClustersWithRelatedNouns()
        if cluster_pickle_file:
            filename = cluster_pickle_file
        else:
            filename = self.cluster_pickle_file
        pickle.dump({'cluster_dict':self.cluster_dict, 'clusters':self.clusters}, open(filename, 'wb'))

        return self.cluster_dict, self.clusters



    def vizClusterWords(self):
        for i,cluster in enumerate(self.clusters):
            print "CLUSTER: ",i
            for word in clusters[cluster]:
                print "-->",word

    def augmentClustersWithRelatedNouns(self):
        new_clusters = {}
        new_cluster_dict = {}
        flattened_cluster_keys = self.clusters.keys()
        flattened_cluster_values = [self.clusters[k] for k in flattened_cluster_keys]
        for i,cluster in enumerate(flattened_cluster_keys):
            cluster_words = list(flattened_cluster_values[i])
            other_words = []
            for x in ([list(s) for s in flattened_cluster_values[:i]]+
                    [list(s) for s in flattened_cluster_values[i+1:]]):
                indices = np.random.randint(len(x), size = 10)
                other_words.extend(np.array(x)[indices])
            related_nouns = self.findRelatedNouns(cluster_words,other_words)
            new_clusters[cluster] = set(list(flattened_cluster_values[i]) + related_nouns)
            for w in new_clusters[cluster]:
                new_cluster_dict[w] = cluster
        self.cluster_dict = new_cluster_dict
        self.clusters = new_clusters



    def findRelatedNouns(self, pos_words, neg_words):
        if not self.model:
            self.model = self.getWordEmbeddingsDict()
        related_word_tuples = self.model.most_similar_cosmul(positive=pos_words,negative=neg_words,topn=100)
        related_words = [w[0] for w in related_word_tuples]
        # #TODO: wn.NOUN is too restrictive?
        # for w in related_words:
            # print "WORD:", w
            # print wn.synsets(w,wn.NOUN)
        related_nouns = [w for w in related_words if len(wn.synsets(w, pos=wn.NOUN))>0]
        return related_nouns

    def vizClusters(self, dump_file='clustered_db_names.p'):
        dataset_words = self.getWordsFromDatasets()
        other_features = {}
        for dataset in self.datasets.find():
            other_features[dataset['_id']] = feature_extractor.FeatureExtractor().extractMetaFeatsFromMongoDoc(dataset)
        dataset_ids = []
        dataset_features = []
        for i,_id in enumerate(dataset_words):
            dataset_ids.append(_id)
            dataset_features.append([0]*self.nclusters)
            for word in dataset_words[_id]:
                if word not in self.cluster_dict:
                    # print word
                    pass
                else:
                    dataset_features[i][self.cluster_dict[word]] += 1
            dataset_features[i].extend(other_features[_id])

        dclusters = MiniBatchKMeans(n_clusters=self.nclusters, max_iter = self.max_iter,
                                        batch_size=self.batch_size,n_init=self.n_init)
        dclusters.fit(dataset_features)
        cluster_dict=defaultdict(list)
        for _id,label in zip(dataset_ids,dclusters.labels_):
            cluster_dict[_id] = label
        dByClusters = {}
        for _id in cluster_dict:
            mongoDataset = self.datasets.find_one({'_id':_id},{'name':True,'description':True})
            if 'name' in mongoDataset:
                clusterKey = mongoDataset['name']
            else:
                clusterKey = _id.split('/')[-1]
            if cluster_dict[_id] in dByClusters:
                dByClusters[cluster_dict[_id]].add((_id,clusterKey))
            else:
                dByClusters[cluster_dict[_id]] = set([(_id,clusterKey)])
        for _cluster in dByClusters:
            dByClusters[_cluster] = sorted(dByClusters[_cluster], key=lambda x: x[1])
            print "----CLUSTER:",_cluster
            for db in dByClusters[_cluster]:
                print db
        pickle.dump(dByClusters, open(dump_file, 'wb'))


if __name__ == '__main__':
    clusterer = Clusterer()
    clusterer.vizClusterWords()

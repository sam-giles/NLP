import math
import nltk
from nltk.corpus import stopwords
from nltk import stem
from string import punctuation
import numpy as np
import re

sw = stopwords.words('english') + list(punctuation) + list("''")
stemmer = stem.PorterStemmer()

##CORPUS READER CLASS
class CorpusReader_TFIDF:

    ##CONSTRUCTOR OBJECT
    def __init__(self, corpus, tf='raw', idf='base', stopword=sw, stemmer=stemmer, ignorecase='yes'):
        self.corpus = corpus
        self.corp_fileids = corpus.fileids()
        self.tf = tf
        self.idf = idf
        self.stopword = stopword
        self.stemmer = stemmer
        self.ignorecase = ignorecase

        #start data processing
        #fileWordCounts represents a list of dictionaries of term occurences for each doc
        #fileWordCounts = {}
        #create vocabulary dict for all words in corpus
        vocabulary = set()
        for fileid in self.corp_fileids:
            #fileWordCounts[fileid] = []
            wordList = self.corpus.words(fileid)
            #wordCount = {}
            #preprocessing vocabulary
            wordList = self.processData(wordList)
            vocabulary.update(wordList)
            #for word in wordList:
            #    if word not in wordCount:
            #        wordCount[word] = 1
            #    else:
            #        wordCount[word] = wordCount[word] + 1
            #fileWordCounts[fileid].append(wordCount)

        vocabulary = list(vocabulary)
        word_index = {word: index for index, word in enumerate(vocabulary)}
        
        vocab_size = len(vocabulary)
        docs_count = len(self.corp_fileids)

        #calculate idf in constructor because that is the same
        #for the entire corpus
        word_idf = np.zeros(vocab_size)
        for fileid in self.corp_fileids:
            wordList = set(self.processData(self.corpus.words(fileid)))
            indexes = [word_index[word] for word in wordList]
            word_idf[indexes] += 1.0
        
        if self.idf is 'smooth':
            word_idf = np.log(docs_count / (1 + word_idf).astype(float))
        else:
            word_idf = np.log(docs_count / (word_idf).astype(float))

        #save important data as part of class instance
        #self.fileWordCounts = fileWordCounts
        self.vocabulary = vocabulary
        self.word_index = word_index
        self.word_idf = word_idf

    ##HELPER FUNCTIONS

    #returns names of the documents in the corpus
    def fileids(self):
        return self.corp_fileids

    #returns raw text within the corpus
    def raw(self, fileids=None):
        if fileids is None:
            fileids = self.corp_fileids
        return self.corpus.raw(fileids)

    #returns the words in list format of the corpus
    def words(self, fileids=None):
        if fileids is None:
            fileids = self.corp_fileids
        return self.corpus.words(fileids)

    #opens the specified input in the corpus
    def open(self, fileid):
        return self.corpus.open(fileid)

    #provides file path of the specified input
    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    ##END HELPER FUNCTIONS

    ##PROGRAM FUNCTIONS

    #return the tf-idf vector corresponding to the input
    def tf_idf(self, fileid=None, filelist=None):
        #first, figure out what data we are using
        if fileid is None:
            if filelist is None:
                filelist = self.corp_fileids
            #else, filelist already exists as a parameter
        else:
            filelist = [fileid]
        #vector will be {filename: [{word: score}, {word: score}, etc.]}
        vector = {}
        for fileid in filelist:
            vector[fileid] = {}
            doc = self.processData(self.corpus.words(fileid))
            #tfidf = tf*idf
            for word in doc:
                vector[fileid][word] = self.wordTF(doc, word) * self.word_idf[self.word_index[word]]
        return vector

    #return the list of the words in the corpus
    def tf_idf_dim(self):
        self.vocabulary.sort()
        return self.vocabulary

    #return a vector corresponding to the tf_idf vector for the new document
    def tf_idf_new(self, words):
        #use same idf from original corpus
        words = self.processData(words)
        vector = {}
        for word in words:
            if word in self.vocabulary:
                vector[word] = self.wordTF(words, word) * self.word_idf[self.word_index[word]]
        return vector

    #return the cosine similarity between two documents in the corpus
    def cosine_sim(self, fileids):
        #first we need the tfidf vector for each document
        tfidf0 = self.tf_idf(fileid=fileids[0])
        tfidf1 = self.tf_idf(fileid=fileids[1])
        #trim out the keys, keep the values
        tfidf0, tfidf1 = tfidf0[fileids[0]], tfidf1[fileids[1]]
        list_0 = list(tfidf0.values())
        list_1 = list(tfidf1.values())

        # convert to numpy arrays
        a = np.array(list_0)
        b = np.array(list_1)

        #make sure the arrays are the same size
        if a.size > b.size:
            b.resize(a.size)
        if b.size > a.size:
            a.resize(b.size)
        
        #do l2 norm
        l2a, l2b = 0.0, 0.0

        #calculate l2a
        for num in a:
            num *= num
            l2a += num
        l2a = math.sqrt(l2a)

        #calculate l2b
        for num in b:
            num *= num
            l2b += num
        l2b = math.sqrt(l2b)

        #adjust the arrays
        a = np.divide(a, l2a)
        b = np.divide(b, l2b)

        #get cosine similarity
        dp = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosineSim = dp / (norm_a * norm_b)

        return cosineSim


    #Theis function returns the cosine similarity between 
    # fileid and the new document specified by the [words] list
    def cosine_sim_new(self, words, fileid):
        #get tfidf of words and fileid
        #tf_idf_new preprocesses words so we don't have to do that here
        words_tfidf = self.tf_idf_new(words)
        file_tfidf = self.tf_idf(fileid=fileid)

        #trim out the key, keep the values
        file_tfidf = file_tfidf[fileid]

        list_words = list(words_tfidf.values())
        list_file = list(file_tfidf.values())

        # convert to numpy arrays
        a = np.array(list_words)
        b = np.array(list_file)

        #make sure the arrays are the same size
        if a.size > b.size:
            b.resize(a.size)
        if b.size > a.size:
            a.resize(b.size)
        
        #do l2 norm
        l2a, l2b = 0.0, 0.0

        #calculate l2a
        for num in a:
            num *= num
            l2a += num
        l2a = math.sqrt(l2a)

        #calculate l2b
        for num in b:
            num *= num
            l2b += num
        l2b = math.sqrt(l2b)

        #adjust the arrays
        a = np.divide(a, l2a)
        b = np.divide(b, l2b)

        #get cosine similarity
        dp = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cosineSim = dp / (norm_a * norm_b)

        return cosineSim

    ##END PROGRAM FUNCTIONS

    ##MY OWN FUNCTIONS

    #my own function called to preprocess the data
    #returns the data after it is processed
    def processData(self, data):
        #now we handle preprocessing
        #check ignorecase
        if self.ignorecase is not 'no':
            data = [w.lower() for w in data]
        #stem before stopwords
        data = [self.stemmer.stem(w) for w in data]
        #check if stemwords are passed in as a file
        sw = []
        if self.stopword != 'none':
            if isinstance(self.stopword, list):
                sw = self.stopword
            else: #it's going to be a filepath
                with open(self.stopword,'r') as f:
                    for line in f:
                        for word in line.split():
                            sw.append(word)
        data = [w for w in data if w not in sw and not w.isdigit()]
        data = [w for w in data if(re.match("^[A-Za-z']*$", w))]
        data = [w.replace("'","") for w in data]
        data = [w for w in data if len(w)>0]
        return data

    #calculates the tf of a given word in a given document
    def wordTF(self, document, word):
        if self.tf is 'raw':
            return float(document.count(word)) / len(document)
        elif self.tf is 'log':
            return float(1 + math.log2(document.count(word)) / len(document))
        else: #self.tf is binary
            if document.count(word) > 0:
                return float(1)
            else:
                return float(0)

    ##END MY FUNCTIONS

##END CORPUS READER CLASS
    

    
    
        



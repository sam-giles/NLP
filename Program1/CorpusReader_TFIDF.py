import math
import nltk
from nltk.corpus import stopwords
from nltk import stem

#Corpus Reader class
class CorpusReader_TFIDF:

    #constructor object
    def __init__(self, corpus, tf='raw', idf='base', stopword='default', stemmer=stem.PorterStemmer, ignorecase='yes'):
        self.corpus = corpus
        self.corp_fileNames = corpus.fileids()
        self.processedWords = self.corpus.words()
        self.tf = tf
        self.idf = idf
        self.stopword = stopword
        self.stemmer = stemmer
        self.ignorecase = ignorecase

    #returns names of the documents in the corpus
    def fileids(self):
        return self.corp_fileNames

    #returns raw text within the corpus
    def raw(self, fileids=None):
        if fileids is None:
            fileids = self.corp_fileNames
        return self.corpus.raw(fileids)

    #returns the words in list format of the corpus
    def words(self, fileids=None):
        if fileids is None:
            fileids = self.corp_fileNames
        return self.corpus.words(fileids)

    #opens the specified input in the corpus
    def open(self, fileid):
        return self.corpus.open(fileid)

    #provides file path of the specified input
    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    #return the tf-idf vector corresponding to the input
    def tf_idf(self, fileid=None, filelist=None):
        #first, figure out what data we are using
        if fileid is None:
            if filelist is None:
                filelist = self.corp_fileNames
            #else, filelist already exists as a parameter
        else:
            filelist = [fileid]
        self.processedWords = self.words(filelist)
        #then, handle any preprocessing
        self.processedWords = self.processData(self.stopword, self.stemmer, self.ignorecase, self.processedWords)
        #then, calculate the tf
        tf = self.calculateTF(self.tf, filelist)
        #then, calculate the idf
        idf = self.calculateIDF(self.idf, filelist)
        #now we have tf and idf, so we just have to calculate tfidf
        tfidf = [[]]
        for index, termVector in enumerate(tf):
            tfidf.append([])
            tfidf[index].append(termVector[0])
            for i in range(len(termVector)):
                #skip the first index, since that's the word itself
                if i > 0:
                    #tfidf = tf * idf
                    tfidf[index].append((termVector[i])*(idf[index][1]))

        return tfidf

    #my own function called to preprocess the data
    def processData(self, stopword, stemmer, ignorecase, data):
        #now we handle preprocessing
        #handle stemming
        data = [stemmer.stem(word) for word in data]

        #if we are ignoring case, make all the words lowercase
        if ignorecase is not 'no':
            data = [word.lower() for word in data]

        #handle stopwords
        if stopword is not 'none':
            sw = []
            if stopword is 'default':
                sw = set(stopwords.words('english'))
            else:
                with open(stopword, 'r') as f:
                    for line in f:
                        for word in line.split():
                            sw.append(word)
                sw = set(sw)
            #if we are ignoring case, we also need to .lower() the stopwords
            if ignorecase is not 'no':
                sw = [word.lower() for word in sw]
            data = [word for word in data if not word in sw]

        #lastly we need to sort the words in alphabetical order and remove duplicates
        data = list(set(data))
        data.sort()
        return data
    
    #my own function, calculates the tf vector and is called in tf_idf
    def calculateTF(self, tf, filelist):
        vector=[[]]
        docs = []
        fileWordList = [[]]
        wordList = []
        #separate out each document that we are using
        for filename in filelist:
            fileWordList = self.words(filename)
            fileWordList = self.processData(self.stopword, self.stemmer, self.ignorecase, fileWordList)
            docs.append(fileWordList)
        for doc in docs:
            for word in doc:
                wordList.append(word)
        for index, word in enumerate(wordList):
            vector.append([])
            #each word is the first index of its list
            vector[index].append(word)
            #now calculate term frequency for each document
            for doc in docs:
                termCount = 0
                for doc_term in doc:
                    print(doc_term, word)
                    if doc_term==word:
                        termCount = termCount+1
                        print(doc_term, word, termCount)
                if tf is 'log':
                    if termCount > 0:
                        termCount = 1 + math.log2(termCount)
                elif tf is 'binary':
                    if termCount > 0:
                        termCount = 1
                    
                vector[index].append(termCount)

        vector = vector[:-1]     
        return vector

    #my own function, calculates the idf vector and is called in tf_idf
    def calculateIDF(self, idf, filelist):
        vector=[[]]
        docs = []
        numDocs = len(filelist)
        #separate out each document that we are using
        for filename in filelist:
            wordList = self.words(filename)
            wordList = self.processData(self.stopword, self.stemmer, self.ignorecase, wordList)
            docs.append(wordList)
        for index, word in enumerate(self.processedWords):
            vector.append([])
            #each word is the first index of its list
            vector[index].append(word)
            n = 0
            for doc in docs:
                if word in doc:
                    n = n+1
            #handle smooth vs base
            if idf is 'smooth':
                idf = math.log2(1+(numDocs/n))
            else:
                idf = math.log2(numDocs/n)
            vector[index].append(idf)
        vector = vector[:-1]
        return vector

    #return the list of the words in the order of the dimension
    # of each corresponding to each vector of the tf-idf vector
    ##TODO: def tf_idf_dim(self):

    #return a vector corresponding to the tf_idf vector for the new document
    ##TODO: def tf_idf_new(self, words):

    #return thecosine similarity between two documents in the corpus
    def cosine_sim(self, fileid):
        tf1 = self.calculateTF(self.tf, [fileid[0]])
        tf2 = self.calculateTF(self.tf, [fileid[1]])
        print(tf2)

    #The function return the cosine similarity between 
    # fileid and the new document specify by the [words] list
    ##TODO: def cosine_sim_new(self, words, fileid):
    

    
    
        



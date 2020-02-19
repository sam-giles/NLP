import nltk
from CorpusReader_TFIDF import CorpusReader_TFIDF
import nltk.corpus
from nltk import stem
import time

start_time = time.time()
ps = stem.PorterStemmer()

brown = nltk.corpus.brown
stateOfTheUnion = nltk.corpus.state_union

brownReader = CorpusReader_TFIDF(brown, tf='raw', idf='base', stopword='default', stemmer=ps, ignorecase='yes')
print(brownReader.cosine_sim(['ca03', 'ca04']))

# stateReader = CorpusReader_TFIDF(stateOfTheUnion, 'tf', 'idf')
# print(stateReader.fileids())

print("---Program took %s seconds ---" % (time.time() - start_time))
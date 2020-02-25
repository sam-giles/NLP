import nltk
from CorpusReader_TFIDF import CorpusReader_TFIDF
import nltk.corpus
from nltk import stem
import time

#start timer
start_time = time.time()

brown = nltk.corpus.brown
stateOfTheUnion = nltk.corpus.state_union

##OUTPUT FOR BROWN
brownReader = CorpusReader_TFIDF(corpus=brown)
print('-----------------------------')
print('Brown Corpus')
print('--------------------')
print('First 15 words of tf_idf_dim')
brown_first15 = brownReader.tf_idf_dim()[:15]
#print the first 15 words of the vector
print(brown_first15)
print('--------------------')
brown_fileids = brownReader.fileids()[:5]
brown_tfidf = brownReader.tf_idf()
#print the scores of those 15 words for the first 5 docs in the corpus
for fileid in brown_fileids:
    print(fileid, end = ' ')
    #tfidf = brownReader.tf_idf(fileid=fileid)
    for word in brown_first15:
        if word not in brown_tfidf[fileid].keys():
            print(0, end = ' ')
        else:
            print(brown_tfidf[fileid][word], end = ' ')
    print()
print('--------------------')
#print cosine similarities
print('ca01 ca02', brownReader.cosine_sim(['ca01', 'ca02']))
print('ca01 ca03', brownReader.cosine_sim(['ca01', 'ca03']))
print('ca01 ca04', brownReader.cosine_sim(['ca01', 'ca04']))
print('ca01 ca05', brownReader.cosine_sim(['ca01', 'ca05']))
print('ca02 cb01', brownReader.cosine_sim(['ca02', 'cb01']))
print('ca02 cb02', brownReader.cosine_sim(['ca02', 'cb02']))
print('ca02 cb03', brownReader.cosine_sim(['ca02', 'cb03']))
print('ca02 cb04', brownReader.cosine_sim(['ca02', 'cb04']))
print('ca02 cb05', brownReader.cosine_sim(['ca02', 'cb05']))
print('ca02 ca02', brownReader.cosine_sim(['ca02', 'ca02']))

##END OUTPUT FOR BROWN

##OUTPUT FOR STATE OF THE UNION

stateReader = CorpusReader_TFIDF(corpus=stateOfTheUnion)
print('-----------------------------')
print('State of the Union Corpus')
print('--------------------')
print('First 15 words of tf_idf_dim')
state_first15 = stateReader.tf_idf_dim()[:15]
#print the first 15 words of the vector
print(state_first15)
print('--------------------')
state_fileids = stateReader.fileids()[:5]
state_tfidf = stateReader.tf_idf()
#print the scores of those 15 words for the first 5 docs in the corpus
for fileid in state_fileids:
    print(fileid, end = ' ')
    #tfidf = brownReader.tf_idf(fileid=fileid)
    for word in state_first15:
        if word not in state_tfidf[fileid].keys():
            print(0, end = ' ')
        else:
            print(state_tfidf[fileid][word], end = ' ')
    print()
print('--------------------')
#print cosine similarities
print('1945-Truman 1946-Truman', stateReader.cosine_sim(['1945-Truman.txt', '1946-Truman.txt']))
print('1945-Truman 1947-Truman', stateReader.cosine_sim(['1945-Truman.txt', '1947-Truman.txt']))
print('1945-Truman 1948-Truman', stateReader.cosine_sim(['1945-Truman.txt', '1948-Truman.txt']))
print('1945-Truman 1949-Truman', stateReader.cosine_sim(['1945-Truman.txt', '1949-Truman.txt']))
print('1945-Truman 1950-Truman', stateReader.cosine_sim(['1945-Truman.txt', '1950-Truman.txt']))
print('2003-GWBush 2004-GWBush', stateReader.cosine_sim(['2003-GWBush.txt', '2004-GWBush.txt']))
print('2003-GWBush 2005-GWBush', stateReader.cosine_sim(['2003-GWBush.txt', '2005-GWBush.txt']))
print('2003-GWBush 2006-GWBush', stateReader.cosine_sim(['2003-GWBush.txt', '2006-GWBush.txt']))
print('2002-GWBush 2006-GWBush', stateReader.cosine_sim(['2002-GWBush.txt', '2006-GWBush.txt']))
print('2003-GWBush 2003-GWBush', stateReader.cosine_sim(['2003-GWBush.txt', '2003-GWBush.txt']))

##END OUTPUT FOR STATE OF THE UNION

print("---Program took %s seconds ---" % (time.time() - start_time))
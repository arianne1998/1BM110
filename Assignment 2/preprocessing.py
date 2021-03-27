from xml.dom import minidom
from xml.etree import cElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import coo_matrix
import math
import fasttext
import fasttext.util

#Clear text files for future saving
open('Data/filteredtext.txt','w').close()

# import xml file and retrieve roots
tree = ET.parse('Data/stackExchange-FAQ.xml')
roots = tree.getroot()

# save xml phrases to text file
for phrases in roots.iter('rephr'):
    appendFile = open('Data/filteredtext.txt', 'a')
    appendFile.write(" " + phrases.text)
    appendFile.close()

# open textfile and read for future processing
file1 = open('Data/filteredtext.txt')
text = file1.read()

# tokenize text file into sentences
sentence_list = sent_tokenize(text)

#tokenize sentences into words
word_list = list()
for i in sentence_list:
    word_list.append(word_tokenize(i))

# Normalization
#removing punctuation and special characters
word_list2 = list()
removetable = str.maketrans('','',"*?.,'<>()")
for i in word_list:
    out_list = [j.translate(removetable) for j in i]
    out_list2 = [x for x in out_list if x]
    word_list2.append(out_list2)

#Convert into lower case
word_list3=list()
for i in word_list2:
    out_list3 = [j.lower() for j in i]
    out_list4 = [x for x in out_list3 if x]
    word_list3.append(out_list4)

#Stopword removal
stop_words = set(stopwords.words('english'))
stop_words.add('nt')
stop_words.add('ca')
stop_words.add('wo')
word_list4=list()
for i in word_list3:
    tokens_without_sw = [word for word in i if not word in stop_words]
    word_list4.append(tokens_without_sw)

# Lemmatizing words
lemmatizer=WordNetLemmatizer()
word_list5=list()
for i in word_list4:
    out_list5=[lemmatizer.lemmatize(j) for j in i]
    word_list5.append(out_list5)

processed_text = word_list5
print(len(word_list5))
df = pd.DataFrame(word_list5)

# Word_list = tokanized list of sentences
# Word_list2 = punctuation and special character removal
# Word_list3 = all letters lower case
# Word_list4 = stopwords removed
# Word_list5 = lemmatized words

########################################################################################################################

# # Create dictionary with question phrase numbers and the words
# processed_textDict = {}
# for i in range(len(processed_text)):
#     processed_textDict[i] = processed_text[i]
#
# # Create dictionary with number of words per question
# numberofwordsDict = {}
# for i in range(len(processed_text)):
#     numberofwordsDict[i] = len(processed_text[i])
#
# # Create dictionary containing unique words and counts
# DF = {}
# for i in range(len(processed_text)):
#     tokens = processed_text[i]
#     for w in tokens:
#         try:
#             DF[w].add(i)
#         except:
#             DF[w]={i}
#
# # Count how many times a word occurs
# for i in DF:
#     DF[i] = len(DF[i])
#
# # Create list of all unique words
# total_vocab = [x for x in DF]

#############################################
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list5_corrected = [" ".join(x) for x in word_list5]

tfidf = tfidf_vectorizer.fit_transform(word_list5_corrected)





#######################################################################
# Probeersels om TF en IDF te berekenen maar nog niet gelukt

def computeTF(wordDict,bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict [word] = count/float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents.keys(),0)
    for document in documents:
        for word, val in document.items():
            if val>0:
                idfDict[word] +=1
    for word,val in idfDict.items():
        idfDict[word] = math.log(N/float(val))
    return idfDict

#idfs = computeIDF(DF)

#print(idfs)
# total_vocab[i] = bagOfWordsi
# numOfWordsi = len(processed_text[i])




############################################
# # Create sparse matrix of text
#
# # Create set with all unique words
# vocab = set()
# n_nonzero = 0
# for questionterms in processed_textDict.values():
#     unique_terms = set(questionterms)
#     vocab |= unique_terms
#     n_nonzero += len(unique_terms)
#
# # make list of question phrase numbers
# questionnames = list(processed_textDict.keys())
#
# # Convert to numpy arrays and set dimensions to variable
# questionnames = np.array(questionnames)
# vocab = np.array(list(vocab))
# vocab_sorter = np.argsort(vocab)
# nquestions = len(questionnames)
# nvocab = len(vocab)
#
# # Create empty numpy arrays of 32 bits
# data = np.empty(n_nonzero,dtype=np.intc)
# rows = np.empty(n_nonzero,dtype=np.intc)
# cols = np.empty(n_nonzero,dtype=np.intc)
#
# ind = 0
# # Go through all questions and question terms
# for questionname, terms in processed_textDict.items():
#     # find indices
#     term_indices = vocab_sorter[np.searchsorted(vocab,terms,sorter=vocab_sorter)]
#     # count the unique terms and get their vocabulary indices
#     uniq_indices,counts = np.unique(term_indices,return_counts = True)
#     # number of unique terms
#     n_vals = len(uniq_indices)
#     # fill slice with data
#     ind_end = ind+n_vals
#     # save counts
#     data[ind:ind_end] = counts
#     # save column index
#     cols[ind:ind_end] = uniq_indices
#     # get index of question in total file
#     doc_idx = np.where(questionnames == questionname)
#     # save as a repeated value
#     rows[ind:ind_end] = np.repeat(doc_idx,n_vals)
#     ind = ind_end
#
# # Create sparse matrix
# dtm = coo_matrix((data, (rows,cols)),shape = (nquestions,nvocab),dtype=np.intc)
# dtm1 = dtm.tocsr()
#
# print(dtm)




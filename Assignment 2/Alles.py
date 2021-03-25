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
import io

#####################################################################
#cleaning
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

# Lemmatizing words
lemmatizer=WordNetLemmatizer()
word_list4=list()
for i in word_list3:
    out_list5=[lemmatizer.lemmatize(j) for j in i]
    word_list4.append(out_list5)

#remove *'s from dataset
lines = word_list4
word_list5 = []
for line in lines:
    if line and line != "*":
        word_list5.append(line)

#Stopword removal
stop_words = set(stopwords.words('english'))
stop_words.add('nt')
stop_words.add('ca')
stop_words.add('wo')
word_list6=list()
for i in word_list5:
    tokens_without_sw = [word for word in i if not word in stop_words]
    word_list6.append(tokens_without_sw)


processed_text = word_list6

df = pd.DataFrame(word_list6)


####################################################################
#text representation and matrix creation TF/IDF
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list6_corrected = [" ".join(x) for x in word_list6]

#vectorize sentence list
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(word_list6_corrected)

#create similarity matrix which shows pairwise similarity, #rows/columns == #sentences
pairwise_similarity = tfidf * tfidf.T
TFIDF_array=pairwise_similarity.toarray()

#fill up diagonal where values are 1
np.fill_diagonal(TFIDF_array, np.nan)

print(TFIDF_array)



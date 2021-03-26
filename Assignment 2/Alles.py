from xml.dom import minidom
from xml.etree import cElementTree as ET

import nltk
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
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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

# open textfile, delete * values and tokenize text file into sentences
data = io.open('Data/filteredtext.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    if line and line != "*":
        questions.append(line)

#tokenize sentences into words
word_list = list()
for i in questions:
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

#Stopword removal
stop_words = set(stopwords.words('english'))
stop_words.add('nt')
stop_words.add('ca')
stop_words.add('wo')
word_list5=list()
for i in word_list4:
    tokens_without_sw = [word for word in i if not word in stop_words]
    word_list5.append(tokens_without_sw)


processed_text = word_list5

df = pd.DataFrame(word_list5)

#################################################################### part B and C WITH FULL PREPROCESSING
####################################################################
#text representation and matrix creation TF/IDF (so part B and C)
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list5_corrected = [" ".join(x) for x in word_list5]

#vectorize sentence list
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_full = vect.fit_transform(word_list5_corrected)

#create similarity matrix which shows pairwise similarity, #rows/columns == #sentences
pairwise_similarity = tfidf_full * tfidf_full.T
TFIDF_array_full=pairwise_similarity.toarray()

#fill up diagonal where values are 1
np.fill_diagonal(TFIDF_array_full, np.nan)

print(TFIDF_array_full)


####################################################################
#data preparation for other 2 models
with open('Data/cleaned_5.txt', 'w') as f:
    for item in word_list5:
        f.write("%s\n" % item)

data = io.open('Data/cleaned_5.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    questions.append(line)

#text representation and matrix creation pre trained model (so part B and C)
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

# Get sentence vectors for first 3 questions
#print all vectors##############################
for i in range(0, len(questions)):
    question = questions[i]
    a_full=ft.get_sentence_vector(question)

# Cosine similarity matrix for pre-trained
vectors = [ft.get_sentence_vector(question) for question in questions]
sim_matrix_pre_full = cosine_similarity(vectors, vectors)

#fill up diagonal where values are 1
np.fill_diagonal(sim_matrix_pre_full, np.nan)

print(sim_matrix_pre_full)

##################################################################
#text representation and matrix creation self trained model (so part B and C)
model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors for first 3 questions
#print all vectors##############################
for i in range(0, len(questions)):
    question = questions[i]
    a_full=model.get_sentence_vector(question)

# Cosine similarity matrix for pre-trained
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self_full = cosine_similarity(vectors, vectors)

#fill up diagonal where values are 1
np.fill_diagonal(sim_matrix_self_full, np.nan)

print(sim_matrix_self_full)

#################################################################### part B and C WITHOUT STOPWORD REMOVAL
####################################################################
#text representation and matrix creation TF/IDF (so part B and C)
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list4_corrected = [" ".join(x) for x in word_list4]

#vectorize sentence list
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_nostop = vect.fit_transform(word_list4_corrected)

#create similarity matrix which shows pairwise similarity, #rows/columns == #sentences
pairwise_similarity = tfidf_nostop * tfidf_nostop.T
TFIDF_array_nostop=pairwise_similarity.toarray()

#fill up diagonal where values are 1
np.fill_diagonal(TFIDF_array_full, np.nan)

print(TFIDF_array_nostop)


####################################################################
#data preparation for other 2 models
with open('Data/cleaned_4.txt', 'w') as f:
    for item in word_list4:
        f.write("%s\n" % item)

data = io.open('Data/cleaned_4.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    questions.append(line)

# Get sentence vectors for first 3 questions
#print all vectors##############################
for i in range(0, len(questions)):
    question = questions[i]
    a_nostop=ft.get_sentence_vector(question)

# Cosine similarity matrix for pre-trained
vectors = [ft.get_sentence_vector(question) for question in questions]
sim_matrix_pre_nostop = cosine_similarity(vectors, vectors)

#fill up diagonal where values are 1
np.fill_diagonal(sim_matrix_pre_nostop, np.nan)

print(sim_matrix_pre_nostop)

##################################################################
#text representation and matrix creation self trained model (so part B and C)
model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors for first 3 questions
#print all vectors##############################
for question in questions:
    a_nostop=model.get_sentence_vector(question)

# Cosine similarity matrix for pre-trained
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self_nostop = cosine_similarity(vectors, vectors)

#fill up diagonal where values are 1
np.fill_diagonal(sim_matrix_self_nostop, 0)

for n in [1,3,5]:
    for question in questions:
        model.test(question, k=n)


print(sim_matrix_self_nostop)

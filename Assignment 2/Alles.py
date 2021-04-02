from xml.dom import minidom
from xml.etree import cElementTree as et
import heapq
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
from lxml import etree


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


#cleaning
#Clear text files for future saving
open('Data/filteredtext.txt','w').close()

# import xml file and retrieve roots
tree = et.parse('Data/stackExchange-FAQ.xml')
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

print(len(word_list))

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
print(len(word_list5))
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
np.fill_diagonal(TFIDF_array_full, 0)



####################################################################
#further data preparation so representations of other 2 models can be made
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
np.fill_diagonal(sim_matrix_pre_full, 0)



##################################################################
#text representation and matrix creation self trained model (so part B and C)
model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors for first 3 questions
#print all vectors##############################
for i in range(0, len(questions)):
    question = questions[i]
    a_full=model.get_sentence_vector(question)

# Cosine similarity matrix for self trained
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self_full = cosine_similarity(vectors, vectors)

#fill up diagonal where values are 1
np.fill_diagonal(sim_matrix_self_full, 0)


#################################################################### part B and C WITHOUT STOPWORD REMOVAL
####################################################################
#text representation and matrix creation TF/IDF
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list4_corrected = [" ".join(x) for x in word_list4]

#vectorize sentence list for representation
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_nostop = vect.fit_transform(word_list4_corrected)

#create similarity matrix showing pairwise similarity based on cosine similarity, #rows/columns == #sentences
pairwise_similarity = tfidf_nostop * tfidf_nostop.T
TFIDF_array_nostop=pairwise_similarity.toarray()

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(TFIDF_array_nostop, 0)



####################################################################
#further data preparation so representations of other 2 models can be made
with open('Data/cleaned_4.txt', 'w') as f:
    for item in word_list4:
        f.write("%s\n" % item)

data = io.open('Data/cleaned_4.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    questions.append(line)

# Get sentence vectors for questions
for i in range(0, len(questions)):
    question = questions[i]
    a_nostop=ft.get_sentence_vector(question)

# create cosine similarity matrix for pre-trained using fasttext
vectors = [ft.get_sentence_vector(question) for question in questions]
sim_matrix_pre_nostop = cosine_similarity(vectors, vectors)

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(sim_matrix_pre_nostop, 0)


##################################################################
#text representation and matrix creation self trained model (so part B and C)
model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors questions
for question in questions:
    a_nostop=model.get_sentence_vector(question)

# Cosine similarity matrix for self trained using fasttext
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self_nostop = cosine_similarity(vectors, vectors)

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(sim_matrix_self_nostop, 0)

#################################################################### part B and C WITHOUT LEMMATISATION AND WIHTOUT STOPWORD REMOVAL
####################################################################
#text representation and matrix creation TF/IDF (so part B and C)
# TF/IDF vectorizer working
tfidf_vectorizer = TfidfVectorizer()
word_list3_corrected = [" ".join(x) for x in word_list3]

#vectorize sentence list
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_nolem = vect.fit_transform(word_list3_corrected)

#create similarity matrix which shows pairwise similarity, #rows/columns == #sentences
pairwise_similarity = tfidf_nolem * tfidf_nolem.T
TFIDF_array_nolem=pairwise_similarity.toarray()

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(TFIDF_array_nolem, 0)



####################################################################
#further data preparation so representations of other 2 models can be made
with open('Data/cleaned_3.txt', 'w') as f:
    for item in word_list3:
        f.write("%s\n" % item)

data = io.open('Data/cleaned_3.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    questions.append(line)

# Get sentence vectors for questions
for i in range(0, len(questions)):
    question = questions[i]
    a_nostop=ft.get_sentence_vector(question)

# Cosine similarity matrix for pre-trained
vectors = [ft.get_sentence_vector(question) for question in questions]
sim_matrix_pre_nolem = cosine_similarity(vectors, vectors)

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(sim_matrix_pre_nolem, 0)


##################################################################
#text representation and matrix creation self trained model (so part B and C)
model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors questions
for question in questions:
    a_nostop=model.get_sentence_vector(question)

# Cosine similarity matrix for self trained
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self_nolem = cosine_similarity(vectors, vectors)

#fill up diagonal where values with 0 for making search of top n in part D easier
np.fill_diagonal(sim_matrix_self_nolem, 0)


################################################################# part D
#create dataframe of questions with their corresponding qapair
df=pd.read_csv("Data/qapairs to csv.csv")
list1=df.iloc[:,0].tolist()
list2=list(range(0,1249))
dict={'qapair':list1,'question':list2}
qapair_df=pd.DataFrame(dict)
print(qapair_df)

#define function that calculates precision based on specified array and top N
def precision_top(array, n):
    #make upper triangular part of array 0 to remove duplicates
    array=np.tril(array)
    #find indices of questions with highest similarity based on given N
    N=n
    a_1d = array.flatten()
    idx_1d = a_1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(idx_1d, array.shape)
    #determine precision and print the performance
    a = 0
    for i in range(0, N):
        if qapair_df[qapair_df['question'] == x_idx[i]]['qapair'].values == qapair_df[qapair_df['question'] == y_idx[i]]['qapair'].values:
            a = a + 1
    precision=a/N
    print("top", N, "precision is:", precision)

#specify all arrays and their top N of which a precision calculation is required
print('precision for fully preprocessed TF/IDF top 5, 3 and 1 respectively are: ')
precision_top(TFIDF_array_full, 5)
precision_top(TFIDF_array_full, 3)
precision_top(TFIDF_array_full, 1)

print('precision for fully preprocessed pretrained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_pre_full, 5)
precision_top(sim_matrix_pre_full, 3)
precision_top(sim_matrix_pre_full, 1)

print('precision for fully preprocessed self trained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_self_full, 5)
precision_top(sim_matrix_self_full, 3)
precision_top(sim_matrix_self_full, 1)
print("\n")

print('precision for data without stopword removal TF/IDF top 5, 3 and 1 respectively are: ')
precision_top(TFIDF_array_nostop, 5)
precision_top(TFIDF_array_nostop, 3)
precision_top(TFIDF_array_nostop, 1)

print('precision for data without stopword removal on pretrained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_pre_nostop, 5)
precision_top(sim_matrix_pre_nostop, 3)
precision_top(sim_matrix_pre_nostop, 1)

print('precision for data without stopword removal on self trained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_self_nostop, 5)
precision_top(sim_matrix_self_nostop, 3)
precision_top(sim_matrix_self_nostop, 1)
print("\n")

print('precision for data without stopword removal and lemmatisation TF/IDF top 5, 3 and 1 respectively are: ')
precision_top(TFIDF_array_nolem, 5)
precision_top(TFIDF_array_nolem, 3)
precision_top(TFIDF_array_nolem, 1)

print('precision for data without stopword removal and lemmatisation on pretrained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_pre_nolem, 5)
precision_top(sim_matrix_pre_nolem, 3)
precision_top(sim_matrix_pre_nolem, 1)

print('precision for data without stopword removal and lemmatisation on self trained using fasttext top 5, 3 and 1 respectively are: ')
precision_top(sim_matrix_self_nolem, 5)
precision_top(sim_matrix_self_nolem, 3)
precision_top(sim_matrix_self_nolem, 1)
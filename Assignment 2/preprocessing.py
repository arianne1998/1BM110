from xml.dom import minidom
from xml.etree import cElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import WordNetLemmatizer

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

# Word_list = tokanized list of sentences
# Word_list2 = punctuation and special character removal
# Word_list3 = all letters lower case
# Word_list4 = stopwords removed
# Word_list5 = lemmatized words

########################################################################################################################

DF = {}
for i in range(len(processed_text)):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w]={i}

print(len(DF))
from xml.dom import minidom
from xml.etree import cElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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

word_list2 = list()
removetable = str.maketrans('','',"*?.,'<>()")
for i in word_list:
    out_list = [j.translate(removetable) for j in i]
    out_list2 = [x for x in out_list if x]
    word_list2.append(out_list2)

print(word_list2)

stop_words = set(stopwords.words('english'))

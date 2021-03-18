from xml.dom import minidom
from xml.etree import cElementTree as ET
from nltk.corpus import stopwords

tree = ET.parse('Data/stackExchange-FAQ.xml')
roots = tree.getroot()

stop_words = set(stopwords.words('english'))
for phrases in roots.iter('rephr'):
    for word in phrases.text:
        if not word in stop_words:
            appendFile = open('Data/filteredtext.txt','a')
            appendFile.write(" "+word)
            appendFile.close()
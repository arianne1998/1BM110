import xml.etree.cElementTree as et
import pandas as pd

tree=et.parse('Data\stackExchange-FAQ.xml')
root=tree.getroot()

qapair=[]
questions=[]


for x in root.iter('qapair'):
    root1=et.Element('root')
    root1=x
    for question in root1.iter('rephr'):
        root2=et.Element('root')
        root2=(question)
        qapair.append(x)
        questions.append(question.text)

df=pd.DataFrame({'QA-pair':qapair, 'question': questions})
print(len(questions))

qapair_int=[]
for i in range(len(qapair)):
    qapair_int.append(i)


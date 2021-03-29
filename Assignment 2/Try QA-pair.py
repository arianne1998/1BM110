import xml.etree.cElementTree as et
import pandas as pd

tree=et.parse('Data\stackExchange-FAQ.xml')
root=tree.getroot()

qapair=[]
questions=[]

questions2=[]
# for child in root:
#     print(child.tag, child.attrib)
#     #questions2.append(question.text)

test=[]

for x in root.iter('qapair'):
    root1=et.Element('root')
    root1=x
    for child in root1:
        test.append('yes')
        if child.tag == 'rephr':
            questions2.append('yes')
        #print(child.tag, child.attrib)
    # for question in root1.iter('rephr'):
    #     root2=et.Element('root')
    #     root2=(question)
    #     qapair.append(x)
    #     questions.append(question.text)

print(len(test))
print(len(questions2))

qapair_set = set(qapair)
qapair_list = list(qapair_set)

qapair_int=[]
for i in range(len(qapair_list)):
    qapair_int.append(i)

dict1={}
for i in qapair_list:
    for k in qapair_int:
        dict1[i]=k
        qapair_int.remove(k)
        break

list1=list()
for i in qapair:
    for k in dict1.keys():
        if k == i:
            list1.append(dict1[k])

dict_qapair={}
for pair in list1:
    for question in questions:
        dict_qapair[pair]=question
        questions.remove(question)
        break

#print(dict_qapair)



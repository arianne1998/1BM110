import io

import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

data = io.open('Data/filteredtext.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    if line and line != "*":
        questions.append(line)

# Get sentence vectors for first 3 questions
for i in range(0, 3):
    question = questions[i]
    print(question)
    print(ft.get_sentence_vector(question))
    print("\n")


model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml')

# Get sentence vectors for first 3 questions
for i in range(0, 3):
    question = questions[i]
    print(question)
    print(model.get_sentence_vector(question))
    print("\n")
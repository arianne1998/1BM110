import io

# Requires: https://visualstudio.microsoft.com/visual-cpp-build-tools/
import fasttext.util
from numpy.random import random
import random
from sklearn.metrics.pairwise import cosine_similarity

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

data = io.open('Data/filteredtext.txt', encoding='utf-8')
lines = data.read().splitlines()
questions = []
for line in lines:
    line = line.strip()
    if line and line != "*":
        questions.append(line)


print(questions)
print(type(questions))


# Get sentence vectors for first 3 questions
for i in range(0, 3):
    question = questions[i]
    print(question)
    print(ft.get_sentence_vector(question))
    print("\n")

model = fasttext.train_unsupervised('Data/stackExchange-FAQ.xml', dim=100)

# Get sentence vectors for first 3 questions
for i in range(0, 3):
    question = questions[i]
    print(question)
    print(model.get_sentence_vector(question))
    print("\n")



# Cosine similarity matrix for pre-trained
vectors = [ft.get_sentence_vector(question) for question in questions]
sim_matrix_pre = cosine_similarity(vectors, vectors)

# Cosine similarity matrix for self trained model
vectors = [model.get_sentence_vector(question) for question in questions]
sim_matrix_self = cosine_similarity(vectors, vectors)


print(sim_matrix_self)
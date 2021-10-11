import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

clean_questions = pd.read_csv("../train.tsv", sep='\t')
# test = pd.read_csv("dev.csv")


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
print(clean_questions.head())
# test["tokens"] = test["text"].apply(tokenizer.tokenize)

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
print("Max sentence length is %s" % max(sentence_lengths))
# test_length = [len(tokens) for tokens in test["tokens"]]


id = []
a = 1
while a<=6977:
    a=a+1
    id.append(a)

print(id)

plt.plot(id, sentence_lengths, label="train_sentence_length")
plt.xlabel("sentence")
plt.ylabel("length")
plt.title("train_sentence_length")
plt.legend()
plt.show()



# id = []
# a = 1
# while a<=5134:
#     a=a+1
#     id.append(a)
#
# print(id)
#
# plt.plot(id, test_length, label="test_sentence_length")
# plt.xlabel("sentence")
# plt.ylabel("length")
# plt.title("test_sentence_length")
# plt.legend()
# plt.show()




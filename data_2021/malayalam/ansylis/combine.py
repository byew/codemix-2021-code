import pandas as pd



train = pd.read_csv('Mal_sentiment_full_train.tsv', sep='\t')
train.to_csv("train" + ".tsv", sep='\t')




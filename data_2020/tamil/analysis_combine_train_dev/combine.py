import pandas as pd

# f1 = pd.read_csv('tamil_train.tsv', sep='\t')
# f2 = pd.read_csv('tamil_dev.tsv', sep='\t')
# file = [f1,f2]
# train = pd.concat(file)
# train.to_csv("train+dev" + ".tsv", sep='\t')

#
# f1 = pd.read_csv('train+dev.tsv', sep='\t')
# f2 = pd.read_csv('tamil_test_answer - tamil_test_answer.tsv', sep='\t')
# file = [f1,f2]
# train = pd.concat(file)
# train.to_csv("train+dev+test" + ".tsv", sep='\t')
#
# train = pd.read_csv('train+dev+test.tsv', sep='\t')
# train.to_csv("train+dev+test" + ".tsv", sep='\t')
#
#
# train = pd.read_csv('train+dev+test.tsv', sep='\t')
# train.to_csv("train+dev+test" + ".tsv", sep='\t',index=False)



train = pd.read_csv('train+dev+test.tsv', sep='\t')
train = train[['id', 'text', 'category']]

train.to_csv("all" + ".tsv", sep='\t',index=False)


from sklearn.utils import shuffle
train = pd.read_csv('all.tsv', sep='\t')
train = shuffle(train)
train.to_csv("all_shuffle" + ".tsv", sep='\t',index=False)

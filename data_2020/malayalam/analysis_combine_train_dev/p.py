import pandas as pd

trail = pd.read_csv('tamil_train.tsv', sep='\t')
trail_d = pd.read_csv('tamil_dev.tsv', sep='\t')


# trail.to_csv('train.csv', index=0)
# trail_d.to_csv('train.csv', index=0)




# trail['category']['not-malayalam'] = 0


# trail[trail['category'].isin(['not-malayalam '])] = 1
# trail[trail['category'].isin(['not-malayalam '])] = 1
# trail[trail['category'].isin(['not-malayalam '])] = 1
# trail[trail['category'].isin(['not-malayalam '])] = 1
# trail[trail['category'].isin(['not-malayalam '])] = 1
trail.loc[trail['category'] =='not-Tamil ', 'category']= 0
trail.loc[trail['category'] =='Positive ', 'category']= 1
trail.loc[trail['category'] =='Negative ', 'category']= 2
trail.loc[trail['category'] =='unknown_state ', 'category']= 3
trail.loc[trail['category'] =='Mixed_feelings ', 'category']= 4

trail_d.loc[trail_d['category'] =='not-Tamil ', 'category']= 0
trail_d.loc[trail_d['category'] =='Positive ', 'category']= 1
trail_d.loc[trail_d['category'] =='Negative ', 'category']= 2
trail_d.loc[trail_d['category'] =='unknown_state ', 'category']= 3
trail_d.loc[trail_d['category'] =='Mixed_feelings ', 'category']= 4



# file = [trail,trail_d]
# train = pd.concat(file)


print(trail)

trail.to_csv('train.csv', index=0)


train = pd.read_csv('train.csv')
train.to_csv('train.csv')



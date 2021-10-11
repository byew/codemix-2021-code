import pandas as pd

trail = pd.read_csv('tamil.csv')
trail_d = pd.read_csv('../../../data_2021/tamil/test.tsv', sep='\t')


# trail.to_csv('train.csv', index=0)
# trail_d.to_csv('train.csv', index=0)




# trail['label']['not-malayalam'] = 0


# ["not-malayalam", "Positive", "Negative", "unknown_state", "Mixed_feelings"]


# trail[trail['label'].isin(['not-malayalam '])] = 1
# trail[trail['label'].isin(['not-malayalam '])] = 1
# trail[trail['label'].isin(['not-malayalam '])] = 1
# trail[trail['label'].isin(['not-malayalam '])] = 1
# trail[trail['label'].isin(['not-malayalam '])] = 1
trail.loc[trail['label'] ==0, 'label']='not-Tamil'
trail.loc[trail['label'] ==1, 'label']= 'Positive'
trail.loc[trail['label'] ==2, 'label']= 'Negative'
trail.loc[trail['label'] ==3, 'label']= 'unknown_state'
trail.loc[trail['label'] ==4, 'label']= 'Mixed_feelings'

# trail_d.loc[trail_d['label'] =='not-Tamil ', 'label']= 0
# trail_d.loc[trail_d['label'] =='Positive ', 'label']= 1
# trail_d.loc[trail_d['label'] =='Negative ', 'label']= 2
# trail_d.loc[trail_d['label'] =='unknown_state ', 'label']= 3
# trail_d.loc[trail_d['label'] =='Mixed_feelings ', 'label']= 4

result = pd.merge(trail_d, trail)


result.to_csv('t_1.tsv',sep='\t' , index=0)




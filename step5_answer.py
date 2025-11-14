import pandas as pd


perd = pd.read_csv('all_predictions.csv')

predict=pd.read_csv('data/acct_predict.csv')

predict['oacct']=predict['acct']
predict['acct']=predict['acct'].apply(lambda x: x[:11])
del predict['label']

data=pd.merge(predict,perd[['acct','label','proba','true_label']],on='acct',how='left')

top_240 = data.nlargest(240, 'proba')

data['label'] = 0  # 先將所有標籤設為 0
# 將選出的 240 個設為 1
data.loc[data['acct'].isin(top_240['acct']), 'label'] = 1

data['label'] = data['label'].astype(int)
del data['acct']
data.rename(columns={'oacct':'acct'},inplace=True)

data[['acct', 'label']].to_csv('submit.csv',index=False)
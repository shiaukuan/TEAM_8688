import pandas as pd
trans=pd.read_csv('data/acct_transaction.csv')

trans['from_acct']=trans['from_acct'].apply(lambda x: x[:11])
trans['to_acct']=trans['to_acct'].apply(lambda x: x[:11])

trans.to_csv('datas/acct_transaction.csv',index=False)
alert=pd.read_csv('data/acct_alert.csv')

alert['acct']=alert['acct'].apply(lambda x: x[:11])
alert.to_csv('datas/acct_alert.csv',index=False)

predict=pd.read_csv('data/acct_predict.csv')
predict['acct']=predict['acct'].apply(lambda x: x[:11])
predict.to_csv('datas/acct_predict.csv',index=False)
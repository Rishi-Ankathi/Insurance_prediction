#1. Load training and testing data
#2. Scaling the training data
#3. save selected data into processed folder

import pandas as pd
import pickle
from data_preprocessing import load_and_split_data
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test = load_and_split_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled).to_csv(r'../data/processed/X_train_scaled.csv',index=False)
pd.DataFrame(X_test_scaled).to_csv(r'../data/processed/X_test_scaled.csv',index=False)
pd.DataFrame(y_train).to_csv(r'../data/processed/y_train.csv',index=False)
pd.DataFrame(y_test).to_csv(r'../data/processed/y_test.csv',index=False)

with open(r'../artifacts/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
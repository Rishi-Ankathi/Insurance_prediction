#1. load processed data from processed folder
#2. create model and train the model
#3. save model in artifacts folder

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv(r'../data/processed/X_train_scaled.csv')
X_test = pd.read_csv(r'../data/processed/X_test_scaled.csv')
y_train = pd.read_csv(r'../data/processed/y_train.csv')
y_test = pd.read_csv(r'../data/processed/y_test.csv')

model = LinearRegression()
model.fit(X_train,y_train)

with open(r'../artifacts/model.pkl','wb') as f:
    pickle.dump(model,f)
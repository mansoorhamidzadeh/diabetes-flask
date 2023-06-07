import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset=pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:,8].values
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)
dataset_scaled = pd.DataFrame(dataset_scaled)
X = dataset_scaled
Y = dataset_Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['Outcome'] )
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)
svc.score(X_test, Y_test)
Y_pred = svc.predict(X_test)
pickle.dump(svc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
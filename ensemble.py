from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score as acs
from catboost import CatBoostClassifier 

dataset = pd.read_csv('D:/PBL-I/dataset/heart_statlog_cleveland_hungary_final.csv')

X = dataset.iloc[:, 0:11]
y = dataset.iloc[:, 11:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


y_train = np.array(y_train)
y_train = y_train.reshape(-1)


y_test = np.array(y_test)
y_test = y_test.reshape(-1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# KNN
kn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
kn.fit(X_train, y_train)


# Decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)


# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# Random forest
rfc = RandomForestClassifier(n_estimators=50, max_depth=15)
rfc.fit(X_train, y_train)

#XgBoost
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

#CatBoost
cbc = CatBoostClassifier()
cbc.fit(X_train,y_train)

#Ada Boost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)

def ensemble(X):
    predictionKN = kn.predict(X)
    predictionDT = dt.predict(X)
    predictionNB = nb.predict(X)
    predictionLR = lr.predict(X)
    predictionRFC = rfc.predict(X)
    #89%
    predictionXG = xgb.predict(X)
    #90%
    predictionCBC = cbc.predict(X)
    #91.5%
    predictionCLF = clf.predict(X)
    #92.5%

    
    ensemble_prediction = np.zeros(len(predictionDT))
    for i in range(len(predictionDT)):
        ensemble_prediction[i] = predictionDT[i]+predictionKN[i] + predictionNB[i]+predictionLR[i]+predictionRFC[i]+predictionXG[i]+predictionCBC[i] + predictionCLF[i]
        

    for j in range(len(predictionDT)):
        if ensemble_prediction[j] > 3:
            ensemble_prediction[j] = 1
        else:
            ensemble_prediction[j] = 0
    return ensemble_prediction


ensemble_prediction=ensemble(X_test)
score = acs(y_test, ensemble_prediction)


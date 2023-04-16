import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score as acs

dataset = pd.read_csv('D:/PBL-I/dataset/heart_statlog_cleveland_hungary_final.csv')

X = dataset.iloc[:,0:11]
y = dataset.iloc[:,11:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)


y_train = np.array(y_train)
y_train = y_train.reshape(-1)


y_test = np.array(y_test)
y_test= y_test.reshape(-1)



#KNN


from sklearn.neighbors import KNeighborsClassifier 

kn = KNeighborsClassifier(n_neighbors = 5,metric='minkowski', p =2)
kn.fit(X_train,y_train)
predictionKN = kn.predict(X_test)
cmkn = confusion_matrix(y_test, predictionKN)
crkn = classification_report(y_test, predictionKN)
askn = acs(y_test, predictionKN)

# Training and testing set from Decision tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
predictionDT = dt.predict(X_test)
cmdt = confusion_matrix(y_test, predictionDT)
crdt = classification_report(y_test, predictionDT)
asdt = acs(y_test, predictionDT)

### Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)
predictionNB = nb.predict(X_test)
cmnb = confusion_matrix(y_test, predictionNB)
crnb = classification_report(y_test, predictionNB)
asnb = acs(y_test, predictionNB)

## Logistic regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictionLR = lr.predict(X_test)
cmlr = confusion_matrix(y_test, predictionLR)
crlr = classification_report(y_test, predictionLR)
aslr = acs(y_test, predictionLR)

##Random forest

from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators= 50,max_depth=15)
rfc.fit(X_train,y_train)
predictionRFC = rfc.predict(X_test)
cmrfc = confusion_matrix(y_test, predictionRFC)
crrfc = classification_report(y_test, predictionRFC)
asrfc = acs(y_test, predictionRFC)

from sklearn.model_selection import cross_val_score


scoreRFC = cross_val_score(rfc,X,y,cv = 10)
k=0
for i in range(0,10):
    k=scoreRFC[i]+k
k=k/10

# Dumping the model

import pickle

data ={"model":rfc}

with open ("D:/PBL-I/Codes/Saved_steps.pkl", 'wb') as file:
    pickle.dump(data,file)
    
    
    
with open ("Saved_steps.pkl", 'rb') as file:
    data = pickle.load(file)

rfc_loaded = data["model"]

pkl_pred=rfc_loaded.predict(X_test)










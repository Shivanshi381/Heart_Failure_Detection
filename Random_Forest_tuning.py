import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score as acs

dataset = pd.read_csv('D:/PBL-I/dataset/heart_statlog_cleveland_hungary_final.csv')

X = dataset.iloc[:,0:11]
y = dataset.iloc[:,11:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

y_train = np.array(y_train)
y_train = y_train.reshape(-1)


y_test = np.array(y_test)
y_test= y_test.reshape(-1)


##Random forest

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(n_estimators= 250,max_depth=15)
rfc.fit(X_train,y_train)
predictionRFC = rfc.predict(X_test)
asrfc = acs(y_test, predictionRFC)




scoreRFC = cross_val_score(rfc,X,y,cv = 10)
k=0
for i in range(0,10):
    k=scoreRFC[i]+k
k=k/10

def FunctionFindBestParams(X_train, y_train):
    
    
    TrialNumber=0
    estimators_list = [50, 100, 250,300]
    max_depth_list = [5, 10 ,15,20]
    
    
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    for estimators_trial in estimators_list:
        for max_depth_trial in max_depth_list:
            TrialNumber+=1
            rfc = RandomForestClassifier(n_estimators=estimators_trial ,max_depth=max_depth_trial)
            rfc.fit(X_train,y_train)
            predictionRFC = rfc.predict(X_test)
            asrfc = acs(y_test, predictionRFC)
            scoreRFC = cross_val_score(rfc,X,y,cv = 10)
            k=0
            for i in range(0,10):
                k=scoreRFC[i]+k
            k=k/10
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','estimator:', estimators_trial,'-', 'max_depth:',max_depth_trial, 'Accuracy:', k)
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber,
                            'estimators'+str(estimators_trial)+'-'+'maxdepth'+str(max_depth_trial), k]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)


final = FunctionFindBestParams(X_train, y_train)










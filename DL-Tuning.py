import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('D:/PBL-I/heart_statlog_cleveland_hungary_final.csv')

X = dataset.iloc[:,0:11]
y = dataset.iloc[:,11:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann= tf.keras.models.Sequential()

def FunctionFindBestParams(X_train, y_train):
    
    
    TrialNumber=0
    batch_size_list=[5, 10, 15, 20]
    epoch_list=[100, 150 ,200]
    
    
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            
            # Creating the classifier ANN model
            ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
            ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
            
            ANN_Model=ann.fit(X_train,y_train, batch_size=batch_size_trial , epochs=epochs_trial, verbose=0)
            # Fetching the accuracy of the training
            Accuracy = ANN_Model.history['accuracy'][-1]
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', Accuracy)
            
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber,
                            'batch_size'+str(batch_size_trial)+'-'+'epoch'+str(epochs_trial), Accuracy]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)
 


ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ANN_Model=ann.fit(X_train,y_train, batch_size=10 , epochs=150, verbose=0)
# Calling the function
ResultsData=FunctionFindBestParams(X_train, y_train)

prediction = ann.predict(X_test)

for i in range(0,238):
    if prediction[i]>0.5:
        prediction[i]=1
    else:
        prediction[i]=0
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score as acs

cm = confusion_matrix(y_test, prediction)
cr = classification_report(y_test, prediction)
ascs = acs(y_test, prediction)

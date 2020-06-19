#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[102]:


full_dataset=pd.read_csv("E:\csv\project1.csv")


# In[103]:


print(full_dataset)


# In[104]:


import matplotlib.pyplot as plt       ## wonderful library for data plotting
full_dataset.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)


# In[105]:


full_dataset[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()


# In[106]:


full_dataset.Pclass.value_counts(normalize=True).plot(kind="bar", alpha = 0.5)


# In[107]:


dataset=full_dataset[['Pclass','Sex','Age','SibSp','Fare','Parch']]


# In[108]:


print(dataset)


# In[109]:


dataset.head()


# ### checking the NAN values in dataset

# In[110]:


dataset.isnull().sum()


# In[111]:


dataset.fillna({'Age':dataset.Age.mean()},inplace=True)


# In[112]:


dataset.isnull().sum()


# In[113]:


data2 = pd.get_dummies(dataset[['SibSp','Age','Parch']])


# In[114]:


data2.head()


# In[115]:


data3 = dataset.select_dtypes(exclude=['object'])


# In[116]:


final_data = pd.concat((data2,data3),axis=1)


# In[117]:


final_data.head()


# In[118]:


final_data.isnull().sum()


# In[119]:


final_data.shape


# In[120]:


y = full_dataset.Survived.values


# In[121]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
xtrain,xtest,ytrain,ytest = train_test_split(final_data.values, y, test_size=0.2,random_state=1)
log = LogisticRegression(C=0.001)
log.fit(xtrain,ytrain)


# In[122]:


log.score(xtest,ytest)
y_pred_log=log.predict(xtest)
print(y_pred_log)
print(log.score(xtest,ytest))


# In[123]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred_log)
print(cm)
acc_logistic_tree = round(log.score(xtrain, ytrain) * 100, 2)
#data.select_dtypes(include=['object'])
## Missing Values


# In[124]:


from sklearn.ensemble import RandomForestClassifier
classifierR=RandomForestClassifier()
classifierR.fit(xtrain,ytrain)
classifierR.score(xtest,ytest)
y_pred_random = classifierR.predict(xtest)
print(y_pred_random)
from sklearn.metrics import confusion_matrix
cmr=confusion_matrix(ytest,y_pred_random)
print(cmr)
acc_randomForest_tree = round(classifierR.score(xtrain, ytrain) * 100, 2)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(xtrain,ytrain)
classifier.score(xtest,ytest)
y_pred_decision = classifier.predict(xtest)
print(y_pred_decision)
from sklearn.metrics import confusion_matrix
cmd=confusion_matrix(ytest,y_pred_decision)
print(cmd)
acc_decision_tree = round(classifier.score(xtrain, ytrain) * 100, 2)


from sklearn.svm import SVC, LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(xtrain, ytrain)
linear_svc.score(xtest,ytest)
Y_pred_svc = linear_svc.predict(xtest)
print(Y_pred_svc)
from sklearn.metrics import confusion_matrix
cm_svc=confusion_matrix(ytest,Y_pred_svc)
print(cm_svc)
acc_svc_tree = round(linear_svc.score(xtrain, ytrain) * 100, 2)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=21, metric="euclidean", p=2)
classifier_knn.fit(xtrain, ytrain)
classifier_knn.score(xtest,ytest)
Y_pred_knn = classifier_knn.predict(xtest)
print(Y_pred_knn)
from sklearn.metrics import confusion_matrix
cm_knn=confusion_matrix(ytest,Y_pred_knn)
print(cm_knn)
acc_knn_tree = round(classifier_knn.score(xtrain, ytrain) * 100, 2)

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Linear SVC',
              'Random Forest', 'Decision Tree'],
    'Score': [acc_logistic_tree, acc_knn_tree,acc_svc_tree,
              acc_randomForest_tree, acc_decision_tree]})
print(models.sort_values(by='Score',ascending=True))


# In[ ]:





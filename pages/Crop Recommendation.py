import streamlit as st

st.title("Crop Recommendation")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.feature_selection
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree

df = pd.read_csv('https://raw.githubusercontent.com/NoName-Stuff/trinit-ai/main/static/crop_recommendation.csv')

st.write(df['label'].value_counts())

fig = plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
st.pyplot(fig)

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

acc = []
model = []

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
st.write("DecisionTrees's Accuracy is: ", x*100)

st.write(classification_report(Ytest,predicted_values))
score = cross_val_score(DecisionTree, features, target,cv=5)

st.write(score)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
st.write("Logistic Regression's Accuracy is: ", x)

st.write(classification_report(Ytest,predicted_values))

score = cross_val_score(LogReg,features,target,cv=5)
st.write(score)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
st.write("RF's Accuracy is: ", x)

st.write(classification_report(Ytest,predicted_values))

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')

accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    st.write (k, '-->', v)
    
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
st.write(prediction)

data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
st.write(prediction)

# Import streamlit application
import streamlit as st

st.title("Tri NIT Hackathon")

# Import required Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import tree

# Load the dataset
def data_load():
    crops_data = pd.read_csv('static/agriculture_data.csv')
    return crops_data

data = data_load()

st.write("Dataset - Overview")
unique_state_rows = data.drop_duplicates(subset=['state'])
st.write(unique_state_rows)

st.write("stats overview")
st.write(data.describe())

st.write("Crops and State overview")
st.write(data['state'].value_counts())

# Plot the heatmap
fig = plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot = True)
st.pyplot(fig)

# Preprocess the data
df = pd.read_csv('static/agriculture_data.csv')
df.dropna(subset=['modal_price'], inplace=True)

# Label-encode the "commodity" column
le = LabelEncoder()
df['commodity'] = le.fit_transform(df['commodity'])

# Select the relevant columns for modeling
X = df[['state', 'district', 'modal_price']]
y = df['commodity']

# One-hot encode the categorical features
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Fit a RandomForestClassifier model to the training data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's accuracy on the test data
accuracy = model.score(X_test, y_test)
st.write(f"Model accuracy: {accuracy:.3f}")

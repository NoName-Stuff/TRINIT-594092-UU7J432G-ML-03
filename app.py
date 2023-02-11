import streamlit as st

st.title("Tri NIT Hackathon")

st.write("Dataset contains the following information")
st.write("State :-")
st.write("District :-")
st.write("Market :-")
st.write("Commodity :-")
st.write("Variety :-")
st.write("arrival_date :-")
st.write("Minimum Price :-")
st.write("Maximum Price :-")
st.write("Modal Price :-")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import tree

def data_load():
    crops_data = pd.read_csv('static/crops.csv')
    return crops_data

data = data_load()

unique_state_rows = data.drop_duplicates(subset=['state'])
st.write(unique_state_rows)

st.write("stats overview")
st.write(data.describe())

st.write("Crops and State overview")
st.write(data['state'].value_counts())

# Preprocess the data
df = data.drop(['State', 'Latitude', 'Longitude', 'Season'], axis=1)
df['Price'] = df['Price'].fillna(df['Price'].mean())
df = pd.get_dummies(df)

# Split the data into training and testing sets
X = df.drop('Soil Type', axis=1)
y = df['Soil Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier on the data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the accuracy of the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Use the trained model to predict the best soil type for a given location, season, and price
location = 'Bihar'
season = 'Kharif'
price = 1500
X_new = pd.DataFrame({
    'Location': [location],
    'Season': [season],
    'Price': [price]
})
X_new = pd.get_dummies(X_new).reindex(columns=X.columns, fill_value=0)
soil_type = rf.predict(X_new)
st.write(f"The best soil type for {location} during {season} at a price of {price} is {soil_type[0]}.")

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

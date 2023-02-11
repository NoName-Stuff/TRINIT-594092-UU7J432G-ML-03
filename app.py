import streamlit as st

st.title("Tri NIT Hackathon")

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

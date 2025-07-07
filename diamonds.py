import pandas as pd 
import streamlit as st

# 1. a) Load the dataset.
data = r"C:\Users\patri\Downloads\diamonds.csv.zip" # I've used an absolute path. Use relative path to access the data.
df = pd.read_csv(data) # Convert csv into dataframe using pandas.


print(df.head(10)) # The data looks good
print(df.shape) # (53940 rows, 11 columns) Good data to work with
print(df.columns) #['Unnamed: 0', 'carat', 'cut', 'color', 'clarity', 'depth', 'table','price', 'x', 'y', 'z']

# 1. b) Explanatory data analysis.
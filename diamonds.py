import pandas as pd 
import streamlit as st


data = r"C:\Users\patri\Downloads\diamonds.csv.zip"
df = pd.read_csv(data)


print(df.head(10))
print(df.shape)
print(df.columns)

print("hi ayden")
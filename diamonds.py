import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

# =================== PAGE CONFIG =================== #
st.set_page_config(page_title="Diamond Price Predictor", page_icon="D", layout="centered")

# =================== LOAD DATA =================== #
df = pd.read_csv("diamonds.csv")
df = df.drop(columns=['Unnamed: 0'])

# =================== SIDEBAR =================== #
with st.sidebar:
     st.image("diamondphoto.png", width=250)
     st.title("Diamond Predictor")
     st.markdown("💎 *Predict your diamond’s value using Machine Learning.*")

     selected = option_menu(
          "",
          ["Home", "Predict", "EDA", "About"],
          icons=["house", "gem", "bar-chart", "info-circle"],
          menu_icon='<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Brilliant-cut_diamond.svg/1024px-Brilliant-cut_diamond.svg.png" width="30">',
          default_index=0,
     )

# =================== CSS STYLING =================== #
st.markdown("""
     <style>
          .big-font {
               font-size:22px !important;
               font-weight:600;
               color:#0e1117;
          }
          .stButton > button {
               background-color: #4CAF50;
               color: white;
               font-size: 18px;
               border-radius: 10px;
               padding: 0.5em 1em;
          }
     </style>
     """, unsafe_allow_html=True)

# =================== HOME TAB =================== #
if selected == "Home":
     st.title(" Diamond Price Predictor")
     st.markdown(
          "Welcome to the Diamond Price Predictor. "
          "This app uses Machine Learning to estimate diamond prices based on carat, height, and color."
     )
     st.image("diamondphoto.png", caption="Diamonds are forever 💍", use_container_width=True)

# =================== PREDICT TAB =================== #
elif selected == "Predict":
     st.title(" Predict Your Diamond's Price")
     st.markdown('<p class="big-font">Enter your diamond\'s features below:</p>', unsafe_allow_html=True)

     color_categories = df['color'].astype('category').cat.categories

     col1, col2 = st.columns(2)
     with col1:
          carat = st.slider("Carat", 0.2, 5.0, 0.5, 0.01)
          color = st.selectbox("Color", list(color_categories))
          color_code = list(color_categories).index(color)
     with col2:
          z = st.slider("Height (z, mm)", 2.0, 6.0, 3.0, 0.1)

     # Prepare model
     df_model = df[['carat', 'z', 'color', 'price']]
     df_model['color'] = df_model['color'].astype('category').cat.codes
     X = df_model[['carat', 'z', 'color']]
     y = df_model['price']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
     model = LinearRegression()
     model.fit(X_train, y_train)

     # Predict
     if st.button(" Predict Price"):
          input_data = pd.DataFrame([[carat, z, color_code]], columns=['carat', 'z', 'color'])
          prediction = model.predict(input_data)[0]
          st.success(f"Estimated price of the diamond: **${prediction:,.2f}**")

# =================== EDA TAB =================== #
elif selected == "EDA":
     st.title("Exploratory Data Analysis")

     # Encode for heatmap
     df_encoded = df.copy()
     for col in ['cut', 'color', 'clarity']:
          df_encoded[col] = df_encoded[col].astype('category').cat.codes

     # Correlation heatmap
     corr = df_encoded.corr()
     fig, ax = plt.subplots(figsize=(10, 8))
     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
     st.pyplot(fig)

     # Feature correlation bar chart
     price_corr = corr['price'].drop('price').sort_values()
     fig2, ax2 = plt.subplots(figsize=(10, 6))
     sns.barplot(x=price_corr, y=price_corr.index, palette='coolwarm', ax=ax2)
     ax2.set_title("Feature Correlation with Price")
     st.pyplot(fig2)

# =================== ABOUT TAB =================== #

elif selected == "About":
     st.title("About This App")

     st.markdown("""
     Welcome to the **Diamond Price Predictor**, a smart web application designed to help you estimate the market value of diamonds using machine learning. Whether you're a gem enthusiast, jeweler, or just curious, this tool simplifies price prediction based on key characteristics like **carat weight**, **color**, and **height (z)**.
     """)

     st.markdown("### What It Does")
     st.markdown("""
     This app uses a **Linear Regression Model** trained on a dataset of over 53,000 real diamonds to provide a near-accurate price estimate. By simply adjusting the input sliders for your diamond’s properties, you’ll get a prediction instantly without needing advanced knowledge in data science.
     """)

     st.markdown("### How It Works")
     st.markdown("""
     - **Machine Learning Algorithm:** Linear Regression  
     - **Dataset:** 53,940 diamond entries from a trusted source  
     - **Features Used:**
          - Carat (weight)
          - Z (height in mm)
          - Color (graded D to J)  
     - **Model Evaluation:**
          - R² Score ≈ 0.86  
          - Root Mean Squared Error ≈ $1,471  
     """)

     st.markdown("### Why It Matters")
     st.markdown("""
     In the real world, diamond prices are influenced by various factors, and estimating their value can be complex. This app uses **data-driven insights** to give an informed estimate whether you're:
     - Shopping or selling diamonds 
     - Learning data science 
     - Exploring EDA and modeling concepts  
     """)

     st.markdown("### Developer Info")
     st.markdown("""
     This Diamond Price Predictor was developed by a passionate team of students from **Strathmore University**:

     - **Ayden Ngnintedem Demanou** — BSc. Statistics & Data Science  
     - **Patrick Dhatemwa** — BSc. Statistics & Data Science  
     - **Ian Paul Mulindwa** — BSc. Statistics & Data Science  
     - **Francis Gitau** — BSc. Statistics & Data Science  
     - **Cynthia Musangi** — BSc. Statistics & Data Science  

     **Mission:** To make machine learning tools accessible and insightful for everyone.  
     **Skills:** Python, Data Analysis, Streamlit, scikit-learn, Data Visualization, Communication  
     """)


     st.markdown("### Tools Used")
     st.markdown("""
     - **Streamlit** – for building the web app interface  
     - **Pandas, NumPy, Seaborn, Matplotlib** – for data handling & visualization  
     - **scikit-learn** – for training the regression model  
     - **HTML & CSS** – for styling and layout  
     """)

     st.markdown("> _“Diamonds are forever, but accurate pricing makes them shine even brighter.”_ ")








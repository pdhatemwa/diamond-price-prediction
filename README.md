# 💎 Diamond Price Predictor – A Machine Learning Web App

Welcome to the **Diamond Price Predictor**, a cloud-based machine learning project that uses regression algorithms to predict the cost of diamonds based on their features.

This project is part of a group assignment following the **data mining process** and integrates a functional, visually appealing web interface.

---

## 📘 Case Study Background

Meet **Stella**, a luxury jewelry store owner in Nairobi. She frequently buys loose diamonds in bulk and wants to make smarter purchasing decisions. However, pricing is inconsistent due to subtle differences in a diamond's cut, size, color, and clarity.

Kumasi Data Lab’s business challenge:  
> _"Can I build a tool that predicts how much a diamond should cost based on its physical characteristics?"_

Our solution:  
> A cloud-hosted application powered by a **linear regression model** that estimates the price of diamonds using key features.

---

## 📦 Dataset Used

We used the popular **Diamonds dataset** (sourced from [ggplot2 in R]), which contains the prices and attributes of over 53,000 diamonds.

**Key facts:**
- **Rows:** 53,940
- **Columns:** 10
- **Target Variable:** `price`
- **Input Features:** `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`

### Sample columns:
| Feature | Description |
|---------|-------------|
| `carat` | Weight of the diamond |
| `cut`   | Quality of the cut (Fair, Good, Very Good, Premium, Ideal) |
| `color` | Diamond color (D-J, with D being best) |
| `clarity` | Measurement of internal flaws |
| `price` | Price in USD |

---

## 🧽 Data Cleaning & Preprocessing

To ensure model accuracy and a clean user experience, we applied the following data processing steps:

- **Dropped irrelevant columns** (like `Unnamed: 0`)
- **Encoded categorical variables** (`cut`, `color`, `clarity`) into numerical codes
- **Handled multicollinearity** by selecting only the strongest independent predictors (`carat`, `z`, `color`)
- **Normalized** the numeric features using standard scaling
- **Split the dataset** into 80% training and 20% testing sets

---

## 🧠 Modeling Approach

### 🧮 Algorithm Used:
- **Linear Regression**

### 🧪 Model Training:
- Train/Test Split: **80/20**
- Evaluation Metrics:
  - **Mean Squared Error (MSE)**: ~2.16M
  - **R² Score**: ~0.86

This means the model explains ~86% of the variability in diamond prices, which is considered strong.

### 📚 Validation:
- We used **holdout validation** and inspected residuals to ensure predictions were unbiased.

---

## 🌐 Web Application

We developed a **cloud-based app** that allows users to input diamond details and instantly receive a predicted price.

### 💻 Tech Stack:
- **Frontend**: Streamlit (with custom styling, fonts, and images for a creative UI)
- **Backend**: Python (Pandas, scikit-learn)
- **Deployment**: Streamlit Cloud / Render / AWS / Heroku *(depending on what you used)*

### 🖼️ Interface Creativity:
- Branded with jewel-tone color palettes
- Diamond icons and clean typography
- Responsive and intuitive layout

### 🔗 Live App:
👉 [Click here to open the Diamond Price Predictor](https://your-app-link.streamlit.app)  
*(Make sure the app is publicly accessible at the time of submission)*

---

## 📊 Presentation Slide Deck

The group also prepared a slide deck summarizing:
- Case study background
- Dataset description
- Feature engineering
- Model performance
- Screenshots of the application
- Key conclusions and next steps

---

## ✅ Conclusion & Business Value

Our diamond price predictor successfully meets the client’s goal:
> Stella can now **confidently estimate diamond prices** before making purchases.

### 🚀 Future Work:
- Add more features like `cut`, `clarity`, and `depth`
- Support batch uploads (e.g. CSV pricing)
- Add a **confidence interval** for predictions
- Integrate pricing trends using external data (like market inflation)

---

## 🧑‍💻 Authors & Collaborators

- Patrick Dhatemwa
- Ayden Demanou
- Ian Mulindwa
- Cynthia Kiilu
- Francis Gitau
- Institution: Strathmore University
- Project: Machine Learning in Practice (Course-Based)

---

## 📁 How to Run This Project Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/diamond-price-predictor.git
   cd diamond-price-predictor  

2. Install the requirements
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py


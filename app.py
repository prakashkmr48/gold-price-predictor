import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Historical Data
years = np.array([...])  # (Include years data from Gold Price Predictor)
prices = np.array([...])  # (Include prices data from Gold Price Predictor)

# Model Training
X = years.reshape(-1, 1)
y = prices
degree = 3
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Streamlit UI
st.title("Gold Price Predictor")
st.write("Enter a future year to predict the gold price.")

year_input = st.number_input("Enter year", min_value=2025, max_value=2100, step=1, value=2030)

if st.button("Predict"):
    year_poly = poly.transform(np.array([[year_input]]))
    predicted_price = model.predict(year_poly)[0]
    st.success(f"Predicted gold price in {year_input}: Rs.{predicted_price:.2f}")

if st.button("Clear"):
    year_input = 2030
    st.experimental_rerun()

st.write("### Gold Price Trend")
fig, ax = plt.subplots()
ax.scatter(years, prices, color='blue', label='Actual Prices')
years_future = np.arange(1964, 2100, 1).reshape(-1, 1)
prices_future = model.predict(poly.transform(years_future))
ax.plot(years_future, prices_future, color='red', label='Prediction')
ax.set_xlabel('Year')
ax.set_ylabel('Gold Price (Rs per 10 grams)')
ax.set_title('Gold Price Prediction')
ax.legend()
st.pyplot(fig)

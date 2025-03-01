import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Data
# Convert years array properly
years = np.array([
    1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 
    1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 
    1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
    2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2011, 2012, 
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 
    2025
], dtype=np.float64).reshape(-1, 1)  # Ensure correct format

prices = np.array([
    63.25, 71.75, 83.75, 102.50, 162.00, 176.00, 184.00, 193.00, 202.00, 
    278.50, 506.00, 540.00, 432.00, 486.00, 685.00, 937.00, 1330.00, 1670.00, 
    1645.00, 1800.00, 1970.00, 2130.00, 2140.00, 2570.00, 3130.00, 3140.00, 
    3200.00, 3466.00, 4334.00, 4140.00, 4598.00, 4680.00, 5160.00, 4725.00, 
    4045.00, 4234.00, 4400.00, 4300.00, 4990.00, 5600.00, 5850.00, 7000.00, 
    10800.00, 12500.00, 14500.00, 18500.00, 26400.00, 31050.00, 29600.00, 
    28006.50, 26343.50, 28623.50, 29667.50, 31438.00, 35220.00, 48651.00, 
    48720.00, 52670.00, 65330.00, 77913.00, 79200.00
], dtype=np.float64)  # Convert to float

# Train a polynomial regression model
degree = 3  
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(years)  # Ensure correct format

model = LinearRegression()
model.fit(X_poly, prices)


# Streamlit UI
st.title("Gold Price Predictor")
st.write("Enter a future year to predict the gold price.")

# User input
year_input = st.number_input("Enter year", min_value=2025, max_value=2100, step=1, value=2030)

if st.button("Predict"):
    year_poly = poly.transform(np.array([[year_input]]))
    predicted_price = model.predict(year_poly)[0]
    st.success(f"Predicted gold price in {year_input}: Rs.{predicted_price:.2f}")

# Clear button
if st.button("Clear"):
    year_input = 2030
    st.experimental_rerun()

# Plot results
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

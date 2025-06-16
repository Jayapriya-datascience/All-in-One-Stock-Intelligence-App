#📘 Project Overview
The **All-in-One Stock Intelligence App** is a powerful, interactive web application designed to simplify and enhance stock market analysis **using machine learning, technical indicators, sentiment analysis,** and** AI-driven insights.**

**Built using Streamlit, this app enables users to:**

Predict future stock prices with LSTM
Visualize key technical analysis charts
Analyze the sentiment of recent news headlines
View company fundamentals like Market Cap, PE Ratio, Revenue
Get instant stock-related answers using Gemini AI

**🎯 Purpose:**
To provide a user-friendly, end-to-end stock analysis dashboard that combines machine learning predictions, financial data, and AI assistance — all in one place.

**👥 Target Audience:**
Finance enthusiasts and retail investors
Data science students & ML learners
Professionals interested in real-world ML + finance projects

#🔍 What This Project Does
This project is an interactive stock market analysis tool that combines:

**1. 📈 Stock Price Prediction**
Uses an LSTM deep learning model to forecast future closing prices based on historical stock data.
Provides performance metrics like RMSE and MAE.
Generates a 7-day price forecast.

**2. 📊 Visual Analytics
Visualizes:**

Actual vs Predicted Close Prices
Simple Moving Averages (SMA-20, SMA-50)
Relative Strength Index (RSI)
MACD & Signal Line

**3. 🧠 News Sentiment Analysis**
Analyzes recent headlines about a stock using TextBlob.
Displays sentiment scores and an average polarity rating.
Visualizes sentiment using a bar chart.

**4. 🧾 Company Fundamentals Viewer**
Retrieves company data like:
Market Cap
PE Ratio
Dividend Yield
EPS
Revenue & Net Income
(via the yfinance API)

**5. 🤖 Gemini AI Assistant**
Users can ask natural language questions about stocks.
Answers generated using the Gemini 1.5 model from Google Generative AI.

**6. 📋 Unified Dashboard**
Combines all features into one dashboard page for a comprehensive market overview.
It’s a complete ML + finance + AI dashboard — perfect for both beginner investors and data science learners. 

#🔁 Project Workflow


#🛠️ Tech Stack – All-in-One Stock Intelligence App
**🧠 Machine Learning / Deep Learning**
TensorFlow / Keras – for building and loading the LSTM model
scikit-learn (joblib) – for scaling data using MinMaxScaler and saving models

**📊 Data Manipulation & Visualization**
Pandas – for data cleaning and manipulation
NumPy – for numerical operations
Matplotlib – for plotting stock trends and indicators

**💻 Web App Development**
Streamlit – for building the interactive web-based user interface
Custom CSS – for styling Streamlit components and layout

**💰 Finance Data APIs**
yFinance – for fetching historical stock prices and company fundamentals

**🧠 Sentiment Analysis**
TextBlob – for analyzing the polarity of stock-related news headlines

**🤖 AI Assistant**
Google Generative AI (Gemini API) – for natural language Q&A about stocks
(gemini-1.5-flash-latest model)

**🗃️ File Management**
scaler.pkl – Pretrained scaler for price data
lstm_stock_model.h5 – Trained LSTM model for price prediction

**🧪 Model Evaluation**
mean_squared_error, mean_absolute_error – from sklearn.metrics

**🔧 Version Control**
Git + GitHub – for source code management and collaboration


#🌟 Features – All-in-One Stock Intelligence App
**🔮 1. LSTM-Based Stock Price Prediction**
Predicts future closing prices using a deep learning LSTM model.
Accepts custom date ranges and stock tickers.
Generates a 7-day forecast with future dates.
Shows evaluation metrics: RMSE and MAE.

**📈 2. Technical Indicator Visualization**
Plots stock data with important technical indicators:
SMA 20 & SMA 50 (Simple Moving Averages)
RSI (Relative Strength Index)
MACD & Signal Line
Clear, interactive charts with labeled plots.

**📊 3. Actual vs Predicted Visualization**
Side-by-side comparison of predicted vs actual stock prices.
Easy to understand performance visualization.

**🧠 4. News Sentiment Analysis**
Uses TextBlob to analyze the sentiment of simulated news headlines.
Shows:
Polarity scores (from -1 to +1)
Emoji indicators (😊, 😐, 😟)
Average sentiment with a summary

Bar chart of sentiment scores

**🧾 5. Company Fundamentals Viewer**
Displays key financial data from yFinance API:
Market Capitalization
PE Ratio
Dividend Yield
EPS
Revenue (TTM)
Net Income

**🤖 6. Gemini AI Q&A Assistant**
Integrates Google Gemini API (gemini-1.5-flash-latest).
Users can ask stock-related questions in plain English.
Displays responses instantly in the sidebar.

**📋 7. Unified Dashboard View
Combines:**
LSTM prediction results
Technical indicators
News sentiment
Company fundamentals
Offers an all-in-one summary panel for easy stock analysis.

**📥 8. Download Forecast**
Allows users to download the 7-day forecast as a CSV file for further analysis or record-keeping.





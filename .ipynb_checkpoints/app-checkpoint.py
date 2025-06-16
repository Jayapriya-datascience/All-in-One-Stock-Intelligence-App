import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from textblob import TextBlob

        


def calculate_indicators(df):
    df = df.copy()

    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


# --- flags & placeholders in session_state ---
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False      # welcome vs main app
for key in ["actual", "predictions", "future_df", "rmse", "mae", "ticker"]:
    st.session_state.setdefault(key, None)
    
st.markdown(
    """
    <style>
    .stApp {                                     /* gradient page bg  */
        background: linear-gradient(135deg,#1f4037,#99f2c8);
        background-attachment: fixed;
    }
    .main .block-container {                     /* semi-transparent panel */
        background: rgba(255,255,255,0.85);
        padding: 2rem 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .welcome-container {                         /* hero card */
        background: rgba(0,0,0,0.35);
        padding: 3rem 1rem;
        border-radius: 20px;
        text-align: center;
        color:#fff;
        margin-top:3vh;
        margin-bottom:4vh;
        box-shadow:0 8px 30px rgba(0,0,0,0.35);
    }
    .logo-img{width:100px;margin-bottom:24px;}
    .welcome-title{font-size:3rem;font-weight:700;}
    .welcome-subtitle{font-size:1.25rem;margin-top:.75rem;font-style:italic;}
    .creator-name{margin-top:1.5rem;font-size:1rem;}
    </style>
    """,
    unsafe_allow_html=True
)
if not st.session_state.show_prediction:
    st.markdown(
        """
        <div class="welcome-container">
            <img src="https://cdn-icons-png.flaticon.com/512/2814/2814578.png"
                 class="logo-img" alt="logo">
            <div class="welcome-title">🎯 All-in-One Stock Intelligence App</div>
            <div class="welcome-subtitle">
                Predict the market like never before with the power of LSTM
            </div>
            <div class="creator-name">
                🔧 Created by <b>Jayapriya</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚀  Start Predicting"):
        st.session_state.show_prediction = True
@st.cache_resource
def load_assets():
    return load_model("lstm_stock_model.h5"), joblib.load("scaler.pkl")
model, scaler = load_assets()

import streamlit as st
import google.generativeai as genai  # ✅ this import is correct

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyBjgnOrnRxEVKY-zguU42x1K5XbEnoki4Y")

# Load the model
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            return "🚫 You’ve reached the quota limit. Try again later or use a lighter model like `gemini-1.5-flash-latest`."
        return f"❌ Gemini error: {e}"



# Streamlit UI
if st.session_state.get("show_prediction", False):
    st.sidebar.title("Navigation")
    st.sidebar.markdown("**Go to:**")
    
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "Predict"

    # Unique keys for each button
    if st.sidebar.button("Predict", key="btn_predict"):
        st.session_state.nav_page = "Predict"
    if st.sidebar.button("Visualization", key="btn_visualization"):
        st.session_state.nav_page = "Visualization"
    if st.sidebar.button("News Sentiment", key="btn_news"):
        st.session_state.nav_page = "News Sentiment"
    if st.sidebar.button("Fundamentals", key="btn_fundamentals"):
        st.session_state.nav_page = "Fundamentals"
    if st.sidebar.button("📋 Dashboard", key="btn_dashboard"):
        st.session_state.nav_page = "📋 Dashboard"

    page = st.session_state.nav_page

    user_input = st.sidebar.text_input("🤖 Ask a question about stocks")
    if user_input:
        st.sidebar.write(f"📤 You asked: {user_input}")
        reply = ask_gemini(user_input)
        st.sidebar.markdown(f"📥 **Answer:** {reply}")



        # Background styling for main app pages

    st.markdown( """ 
        <style>
        .stApp { 
        background-image: url('https://images.unsplash.com/photo-1518770660439-4636190af475'); 
        background-size: cover; 
        background-position: center; 
        background-repeat: no-repeat; 
        background-attachment: fixed; 
        } 
        .main .block-container { 
        background-color: rgba(255, 255, 255, 0.9); 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        } 
        h1, h2, h3, h4 { color: #003366; } 
        .stButton>button { background-color: #1f77b4; 
        color: white; 
        border-radius: 8px; 
        padding: 0.5rem 1rem; 
        border: none;
        } 
        .stSidebar { background-color: rgba(255, 255, 255, 0.9);
        } 
        </style>
        """, unsafe_allow_html=True )



        # ───────── Predict page ─────────
    if page == "Predict":
        st.header("📈 All-in-One Stock Intelligence App")
        st.write("Predict future stock prices based on historical data using your trained LSTM model.")

        # user inputs
        ticker = st.text_input(
            "Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS, GOOGL, AMZN, TCS.NS, RELIANCE.NS):",
            value="AAPL"
        )
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
        with col2:
            end_date   = st.date_input("End Date",   value=pd.to_datetime("2023-01-01"))

        # predict button
        if st.button("Predict"):
            st.info("⏳ Downloading stock data…")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                st.warning("⚠️ No data found. Check ticker or date range."); st.stop()
            if len(df) < 70:
                st.warning(f"⚠️ Only {len(df)} rows; need ≥70."); st.stop()

            st.success(f"✅ Loaded {len(df)} records for {ticker}")

            # prepare sequences
            close = df[["Close"]]
            scaled = scaler.transform(close)
            seq_len = 60
            X = np.array([scaled[i-seq_len:i,0] for i in range(seq_len, len(scaled))])
            X = X.reshape(-1, seq_len, 1)

            # predict
            preds = scaler.inverse_transform(model.predict(X))
            actual = close[seq_len:].values

            # metrics
            mse  = mean_squared_error(actual, preds)
            rmse = math.sqrt(mse)
            mae  = mean_absolute_error(actual, preds)

            # 7-day forecast
            n_days = 7
            last_seq = scaled[-seq_len:]
            fut_scaled = []
            for _ in range(n_days):
                p = model.predict(last_seq.reshape(1, seq_len, 1))[0,0]
                fut_scaled.append(p)
                last_seq = np.append(last_seq, [[p]], axis=0)[-seq_len:]
            future = scaler.inverse_transform(np.array(fut_scaled).reshape(-1,1)).flatten()
            future_dates = pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=n_days)
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Close Price": future})

            # save for visualization
            st.session_state.actual      = actual.flatten()
            st.session_state.predictions = preds.flatten()
            st.session_state.rmse        = rmse
            st.session_state.mae         = mae
            st.session_state.future_df   = future_df
            st.session_state.ticker      = ticker

            st.success("✅ Prediction complete! Open the **Visualization** page.")

            
    # ───────── Visualization page ─────────
    if page == "Visualization":

        if st.session_state.future_df is None:
            st.warning("Run a prediction first on the **Predict** page."); st.stop()

        ticker = st.session_state.ticker
        st.header(f"📊 Visualization for {ticker}")

        # 1) Actual vs predicted
        st.subheader("📉 Actual vs Predicted Close Price")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.session_state.actual,      label="Actual",    linewidth=2)
        ax.plot(st.session_state.predictions, label="Predicted", linewidth=2)
        ax.set_xlabel("Time"); ax.set_ylabel("Price (USD)"); ax.legend()
        st.pyplot(fig)

        # 2) Metrics
        st.subheader("📈 Model Performance")
        st.metric("RMSE", f"{st.session_state.rmse:.2f}")
        st.metric("MAE",  f"{st.session_state.mae:.2f}")

        # 3) Forecast table
        st.subheader("🔮 7-Day Forecast")
        st.dataframe(st.session_state.future_df)

        # 4) CSV download
        csv = st.session_state.future_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Forecast as CSV",
            csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )
        # 5) Technical Indicators
        st.subheader("📊 Technical Indicators (SMA / RSI / MACD)")
        df = yf.download(ticker, start="2020-01-01", end=pd.to_datetime("today"), progress=False)
        df = calculate_indicators(df)

        tab1, tab2, tab3 = st.tabs(["📉 SMA", "📈 RSI", "📉 MACD"])

        with tab1:
            st.write("### Simple Moving Averages (20-day & 50-day)")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Close'], label="Close Price", linewidth=1.5)
            ax.plot(df['SMA20'], label="SMA 20", linestyle="--")
            ax.plot(df['SMA50'], label="SMA 50", linestyle="--")
            ax.legend()
            ax.set_title("SMA Overlay")
            st.pyplot(fig)

        with tab2:
            st.write("### Relative Strength Index (RSI)")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(df['RSI'], color="purple")
            ax.axhline(70, color="red", linestyle="--")
            ax.axhline(30, color="green", linestyle="--")
            ax.set_title("RSI (14-day)")
            st.pyplot(fig)

        with tab3:
            st.write("### MACD and Signal Line")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['MACD'], label="MACD", color="blue")
            ax.plot(df['Signal'], label="Signal", color="orange")
            ax.axhline(0, color="gray", linestyle="--")
            ax.legend()
            ax.set_title("MACD Indicator")
            st.pyplot(fig)



    elif page == "News Sentiment":
        st.header("🧠 News Sentiment Analysis")
        ticker = st.session_state.get("ticker") or st.text_input("Enter a Stock Ticker to Analyze Sentiment", "AAPL")
        if not ticker:
            st.warning("Please enter or run prediction to select a stock ticker.")
            st.stop()

        st.info(f"Fetching news headlines related to **{ticker}**...")

    # Simulated sample headlines (Replace this later with real API)
        news_list = [
        f"{ticker} stock surges after positive earnings report",
        f"{ticker} faces regulatory challenges amid new policies",
        f"Experts bullish on {ticker}'s future in AI sector",
        f"{ticker} hits 52-week low amidst market turmoil",
        f"Investors show mixed reactions to {ticker} quarterly results"
      ]

        from textblob import TextBlob
        sentiment_scores = [TextBlob(news).sentiment.polarity for news in news_list]
        avg_sentiment = np.mean(sentiment_scores)

        st.subheader("🗞️ Sample News Headlines:")
        for i, news in enumerate(news_list):
            score = sentiment_scores[i]
            emoji = "😊" if score > 0 else "😐" if score == 0 else "😟"
            st.markdown(f"- {news}  &nbsp;&nbsp;&nbsp;**Sentiment:** `{score:.2f}` {emoji}")

        st.subheader("📊 Summary")
        sentiment_text = (
        "😊 Positive Outlook" if avg_sentiment > 0.2 else
        "😐 Neutral Outlook" if -0.2 <= avg_sentiment <= 0.2 else
        "😟 Negative Outlook"
        )
        st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}", sentiment_text)

        st.subheader("📈 Sentiment Score Visualization")
        fig, ax = plt.subplots()
        ax.bar(range(len(news_list)), sentiment_scores, color='skyblue')
        ax.set_xticks(range(len(news_list)))
        ax.set_xticklabels([f"News {i+1}" for i in range(len(news_list))], rotation=45)
        ax.set_ylabel("Sentiment Polarity")
        st.pyplot(fig)
    
    elif page == "Fundamentals":
       st.header("🧾 Company Fundamentals")
       ticker = st.text_input("Enter Stock Ticker for Fundamentals (e.g. AAPL, TCS.NS):", value=st.session_state.ticker or "AAPL")

       def format_large_number(n):
           if not n:
               return "N/A"
           elif n >= 1_000_000_000:
               return f"${n / 1_000_000_000:.2f}B"
           elif n >= 1_000_000:
               return f"${n / 1_000_000:.2f}M"
           else:
               return f"${n:,.2f}"
       if ticker:
           try:
               stock = yf.Ticker(ticker)
               info = stock.info
               st.write("📊 **Fundamental Stats**")
               st.write({
                "Market Cap": format_large_number(info.get("marketCap")),
                "PE Ratio": info.get("trailingPE") or "N/A",
                "Dividend Yield": f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "N/A",
                "EPS": info.get("trailingEps") or "N/A",
                "Revenue (TTM)": format_large_number(info.get("totalRevenue")),
                "Net Income (TTM)": format_large_number(info.get("netIncomeToCommon")),
                })
           except Exception as e:
               st.error(f"❌ Failed to fetch fundamentals for {ticker}: {e}")

    elif page == "📋 Dashboard":
        st.title("📋 Unified Stock Dashboard")

        ticker = st.session_state.get("ticker") or st.text_input("Enter Stock Ticker", "AAPL")

        # Load and prepare data
        df = yf.download(ticker, start="2020-01-01", end=pd.to_datetime("today"), progress=False)
        df = calculate_indicators(df)

        # --- Section 1: Price Prediction Summary ---
        st.header("📈 LSTM Price Prediction Summary")
        if st.session_state.predictions is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{st.session_state.rmse:.2f}")
                st.metric("MAE", f"{st.session_state.mae:.2f}")
            with col2:
                st.write("### 7-Day Forecast")
                st.dataframe(st.session_state.future_df)
        else:
            st.warning("❗ Run prediction from 'Predict' tab to view summary.")

        # --- Section 2: SMA, RSI, MACD ---
        st.header("📊 Technical Indicators")

        tab1, tab2, tab3 = st.tabs(["SMA", "RSI", "MACD"])
        with tab1:
            st.write("### Simple Moving Averages")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Close'], label="Close Price", linewidth=1.5)
            ax.plot(df['SMA20'], label="SMA 20", linestyle="--")
            ax.plot(df['SMA50'], label="SMA 50", linestyle="--")
            ax.legend(); st.pyplot(fig)
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df['RSI'], color="purple")
            ax.axhline(70, color="red", linestyle="--")
            ax.axhline(30, color="green", linestyle="--")
            ax.set_title("RSI"); st.pyplot(fig)
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df['MACD'], label="MACD", color="blue")
            ax.plot(df['Signal'], label="Signal", color="orange")
            ax.axhline(0, color="gray", linestyle="--")
            ax.legend(); st.pyplot(fig)

        # --- Section 3: News Sentiment ---
        st.header("🧠 News Sentiment Analysis")
        news_list = [
            f"{ticker} stock surges after positive earnings report",
            f"{ticker} faces regulatory challenges amid new policies",
            f"Experts bullish on {ticker}'s future in AI sector",
            f"{ticker} hits 52-week low amidst market turmoil",
            f"Investors show mixed reactions to {ticker} quarterly results"
        ]
        sentiment_scores = [TextBlob(news).sentiment.polarity for news in news_list]
        avg_sentiment = np.mean(sentiment_scores)

        for i, news in enumerate(news_list):
            score = sentiment_scores[i]
            emoji = "😊" if score > 0 else "😐" if score == 0 else "😟"
            st.markdown(f"- {news}  &nbsp;&nbsp;&nbsp;**Sentiment:** `{score:.2f}` {emoji}")

        st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")

                # --- Section 4: Company Fundamentals ---
        st.header("🧾 Company Fundamentals")

        def format_large_number(n):
            if not n:
                return "N/A"
            elif n >= 1_000_000_000:
                return f"${n / 1_000_000_000:.2f}B"
            elif n >= 1_000_000:
                return f"${n / 1_000_000:.2f}M"
            else:
                return f"${n:,.2f}"

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                "Market Cap": format_large_number(info.get("marketCap")),
                "PE Ratio": info.get("trailingPE") or "N/A",
                "Dividend Yield": f"{info.get('dividendYield') * 100:.2f}%" if info.get("dividendYield") else "N/A",
                "EPS": info.get("trailingEps") or "N/A",
                "Revenue (TTM)": format_large_number(info.get("totalRevenue")),
                "Net Income (TTM)": format_large_number(info.get("netIncomeToCommon")),
            }

            for key, value in fundamentals.items():
                st.write(f"**{key}:** {value}")

        except Exception as e:
            st.error(f"❌ Failed to fetch fundamentals for {ticker}: {e}")





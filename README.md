# 📈 Multi-Step Stock Market Forecasting — ANN vs RNN

A deep learning project that compares **Artificial Neural Network (ANN)** and **Recurrent Neural Network (RNN)** for multi-step stock price prediction, with an interactive Streamlit dashboard.

> **⚠️ Disclaimer:** This project is for educational purposes only. Do NOT make financial decisions based on these predictions.

## 🌐 Live Demo

🔗 **[Try the app live on Streamlit Cloud](https://YOUR_APP_NAME.streamlit.app)**

> Replace the link above with your actual Streamlit Cloud URL after deployment.

---

## 🎯 What This Project Does

- Downloads **11 years** of real stock data (2015–2026) from Yahoo Finance
- Engineers **14 technical indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Trains two neural networks — **ANN** and **SimpleRNN** — using PyTorch
- Compares their performance with 5 evaluation metrics
- Presents everything in a **dark-themed interactive Streamlit dashboard**

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

---

## 🏗️ How It Works

```
Yahoo Finance API
       │
       ▼
 Raw Stock Data (Open, High, Low, Close, Volume)
       │
       ▼
 + 14 Technical Indicators (SMA, EMA, RSI, MACD, etc.)
       │
       ▼
 Scale to [0,1] → Sliding Windows (60 days in → 5 days out)
       │
       ├──────────────┐
       ▼              ▼
      ANN            RNN
   (Flatten        (Step-by-step
    + Dense)        processing)
       │              │
       ▼              ▼
   5 predicted    5 predicted
   close prices   close prices
       │              │
       └──────┬───────┘
              ▼
       Compare Metrics
        🏆 Winner
```

---

## 🧠 Model Architectures

### ANN (Feed-Forward)

Flattens the 60-day window into one long vector. Fast but **loses time ordering**.

```
Input (60×19) → Flatten → Dense(128) → ReLU → Dropout
→ Dense(64) → ReLU → Dropout → Dense(32) → ReLU → Dense(5)
```

### RNN (Recurrent)

Reads one day at a time, carrying memory via hidden state. **Understands time ordering**.

```
Input (60×19) → RNN(hidden=64, layers=2) → Last Hidden State → Dense(5)
```

---

## 📊 Technical Indicators Used

- **SMA** (20, 50) — Simple Moving Average
- **EMA** (12, 26) — Exponential Moving Average
- **MACD** — Moving Average Convergence Divergence (line + signal + histogram)
- **RSI** (14) — Relative Strength Index
- **Bollinger Bands** — Upper, Middle, Lower bands
- **ATR** (14) — Average True Range
- **Daily Return** — Percentage change
- **Log Return** — Logarithmic return

---

## 🖥️ Dashboard Features

The app has a **dark glassmorphism theme** with 4 tabs:

**📊 Data Explorer** — Candlestick charts, volume bars, summary stats, CSV download

**📈 Technical Indicators** — Toggle indicators on/off, RSI & MACD panels, indicator guide

**🤖 Training & Predictions** — Model architecture display, live training progress, loss curves, metrics comparison table, winner banner, actual vs predicted charts, scatter plot, CSV download

**🔮 Forecast** — Predict next N business days, both models side by side, CSV download

**Sidebar Controls** — Stock picker (dropdown or custom), date range, lookback window, forecast horizon, epochs, batch size, learning rate

---

## 📈 Evaluation Metrics

The models are compared on 5 metrics (lower is better, except R²):

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error (main winner metric)
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination (higher is better, max 1.0)
- **MAPE** — Mean Absolute Percentage Error

---

## 📦 Tech Stack

- **Data**: `yfinance`, `pandas`, `numpy`
- **Indicators**: `ta` (Technical Analysis library)
- **Deep Learning**: `PyTorch`
- **Metrics**: `scikit-learn`
- **Charts**: `plotly`
- **Dashboard**: `streamlit`

---

## 🔮 What's Next — Upgrading to LSTM / GRU

The SimpleRNN struggles with long sequences (vanishing gradient problem). Upgrading to LSTM or GRU is a **one-line change**:

```python
# Current (SimpleRNN)
self.rnn = nn.RNN(input_size, hidden_size, num_layers)

# Upgrade to LSTM
self.rnn = nn.LSTM(input_size, hidden_size, num_layers)

# Upgrade to GRU
self.rnn = nn.GRU(input_size, hidden_size, num_layers)
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

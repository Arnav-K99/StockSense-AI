import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Market Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Dark Glassmorphism Theme
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 40, 0.9);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(78, 205, 196, 0.2);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin-bottom: 16px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(78,205,196,0.15), rgba(255,107,107,0.1));
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(78,205,196,0.2);
        padding: 20px;
        text-align: center;
    }
    
    .winner-banner {
        background: linear-gradient(135deg, rgba(78,205,196,0.3), rgba(255,230,109,0.2));
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(78,205,196,0.4);
        padding: 20px;
        text-align: center;
        font-size: 1.3em;
        font-weight: 600;
        margin: 16px 0;
    }
    
    h1, h2, h3 {
        background: linear-gradient(90deg, #4ECDC4, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(78,205,196,0.3), rgba(255,107,107,0.2));
        border: 1px solid rgba(78,205,196,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Fetching stock data...")
def fetch_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    if data.empty:
        raise ValueError(f"No data found for '{ticker}'.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna().sort_index()
    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data(show_spinner="Computing technical indicators...")
def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df = df.dropna()
    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df, lookback, forecast_horizon):
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                       'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                       'MACD', 'MACD_Signal', 'MACD_Histogram',
                       'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                       'ATR', 'Daily_Return', 'Log_Return']
    close_index = feature_columns.index('Close')
    data_selected = df[feature_columns].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_selected)

    X, y = [], []
    for i in range(lookback, len(data_scaled) - forecast_horizon + 1):
        X.append(data_scaled[i - lookback: i])
        y.append(data_scaled[i: i + forecast_horizon, close_index])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
            torch.FloatTensor(y_train), torch.FloatTensor(y_test),
            scaler, close_index, len(feature_columns))


# ============================================================
# MODELS
# ============================================================

class StockANN(nn.Module):
    def __init__(self, lookback, input_size, forecast_horizon):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )
    def forward(self, x):
        return self.network(x)


class StockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, forecast_horizon)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        return self.fc(out)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'val_loss': []}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(loader)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.6f} | Val: {val_loss:.6f}")

    progress_bar.empty()
    status_text.empty()
    return history


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    actuals = y_test.numpy()
    mse = mean_squared_error(actuals, preds)
    return {
        'MSE': round(mse, 6),
        'RMSE': round(np.sqrt(mse), 6),
        'MAE': round(mean_absolute_error(actuals, preds), 6),
        'R²': round(r2_score(actuals, preds), 4),
        'MAPE (%)': round(np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100, 2)
    }, preds


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🎯 Configuration")

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NFLX"]
ticker_mode = st.sidebar.radio("Choose stock by", ["Popular Stocks", "Custom Ticker"], horizontal=True)

if ticker_mode == "Popular Stocks":
    TICKER = st.sidebar.selectbox("Select Stock", TICKERS, index=0)
else:
    TICKER = st.sidebar.text_input("Enter ticker symbol", "NVDA").upper()

START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
END_DATE = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-06-12"))

LOOKBACK = st.sidebar.slider("Lookback Window (days)", 30, 120, 60)
FORECAST_HORIZON = st.sidebar.slider("Forecast Horizon (days)", 1, 10, 5)
EPOCHS = st.sidebar.slider("Training Epochs", 10, 100, 50)
BATCH_SIZE = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
LEARNING_RATE = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

train_button = st.sidebar.button("🚀 Train Models", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.markdown("ANN vs RNN stock forecaster. Built for learning — **not financial advice**.")

# ============================================================
# HEADER
# ============================================================

st.markdown("# 📈 Stock Market Forecaster")
st.markdown(f"##### Comparing **ANN** vs **RNN** on `{TICKER}`")

# ============================================================
# LOAD DATA
# ============================================================

try:
    raw_data = fetch_stock_data(TICKER, str(START_DATE), str(END_DATE))
    df = add_technical_indicators(raw_data)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "📈 Technical Indicators",
    "🤖 Training & Predictions",
    "🔮 Forecast"
])

# ============================================================
# TAB 1: DATA EXPLORER
# ============================================================

with tab1:
    st.markdown("## 📊 Data Explorer")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trading Days", f"{len(raw_data):,}")
    col2.metric("Latest Close", f"${raw_data['Close'].iloc[-1]:.2f}")
    col3.metric("All-Time High", f"${raw_data['Close'].max():.2f}")
    col4.metric("All-Time Low", f"${raw_data['Close'].min():.2f}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.7, 0.3], subplot_titles=('Price', 'Volume'))
    fig.add_trace(go.Candlestick(
        x=raw_data.index, open=raw_data['Open'], high=raw_data['High'],
        low=raw_data['Low'], close=raw_data['Close'], name='Price'
    ), row=1, col=1)
    vol_colors = ['#FF6B6B' if raw_data['Close'].iloc[i] < raw_data['Open'].iloc[i]
                  else '#4ECDC4' for i in range(len(raw_data))]
    fig.add_trace(go.Bar(x=raw_data.index, y=raw_data['Volume'],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 View Raw Data Table"):
        st.dataframe(raw_data, use_container_width=True)

    csv = raw_data.to_csv()
    st.download_button("📥 Download Raw Data (CSV)", csv, f"{TICKER}_raw_data.csv", "text/csv")


# ============================================================
# TAB 2: TECHNICAL INDICATORS
# ============================================================

with tab2:
    st.markdown("## 📈 Technical Indicators")

    st.markdown("**Toggle indicators to display:**")
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    show_sma = ind_col1.checkbox("SMA (20, 50)", value=True)
    show_ema = ind_col1.checkbox("EMA (12, 26)", value=False)
    show_bb = ind_col2.checkbox("Bollinger Bands", value=True)
    show_rsi = ind_col2.checkbox("RSI", value=True)
    show_macd = ind_col3.checkbox("MACD", value=True)

    num_panels = 1 + int(show_rsi) + int(show_macd)
    heights = [0.5] + [0.25] * (num_panels - 1) if num_panels > 1 else [1.0]
    titles = ['Price'] + (['RSI'] if show_rsi else []) + (['MACD'] if show_macd else [])
    fig = make_subplots(rows=num_panels, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=heights, subplot_titles=titles)

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close',
                             line=dict(color='white', width=1.5)), row=1, col=1)
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                                 line=dict(color='#FF6B6B', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                                 line=dict(color='#FFE66D', width=1.2)), row=1, col=1)
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12',
                                 line=dict(color='#95E1D3', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_26'], name='EMA 26',
                                 line=dict(color='#F38181', width=1.2)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                 line=dict(color='rgba(173,216,230,0.5)', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                 line=dict(color='rgba(173,216,230,0.5)', dash='dash'),
                                 fill='tonexty', fillcolor='rgba(173,216,230,0.07)'), row=1, col=1)

    current_row = 2
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                 line=dict(color='#FFE66D', width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                 line=dict(color='#FF6B6B', width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                 line=dict(color='#4ECDC4', width=1.5)), row=current_row, col=1)
        hist_colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in df['MACD_Histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                             marker_color=hist_colors), row=current_row, col=1)

    fig.update_layout(height=200 + num_panels * 250, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📘 Indicator Guide"):
        st.markdown("""
| Indicator | What It Measures | How To Read |
|-----------|-----------------|-------------|
| **SMA** | Average price over N days | Price above SMA = uptrend |
| **EMA** | Weighted recent average | Reacts faster than SMA to changes |
| **RSI** | Momentum (0-100) | Above 70 = overbought, Below 30 = oversold |
| **MACD** | Trend strength & direction | MACD crossing above Signal = bullish |
| **Bollinger Bands** | Volatility envelope | Price touching bands = potential reversal |
        """)


# ============================================================
# TAB 3: TRAINING & PREDICTIONS
# ============================================================

with tab3:
    st.markdown("## 🤖 Training & Predictions")

    if train_button:
        st.markdown("### ⚙️ Preprocessing...")
        X_train, X_test, y_train, y_test, scaler, close_idx, num_features = preprocess_data(df, LOOKBACK, FORECAST_HORIZON)
        st.success(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples | Features: {num_features}")

        ann_model = StockANN(LOOKBACK, num_features, FORECAST_HORIZON)
        rnn_model = StockRNN(num_features, hidden_size=64, num_layers=2, forecast_horizon=FORECAST_HORIZON)

        st.markdown("### 🏗️ Model Architectures")
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown("**ANN (Feed-Forward)**")
            ann_params = sum(p.numel() for p in ann_model.parameters())
            st.code(f"Flatten → Linear({LOOKBACK * num_features}→128) → ReLU → Dropout\n"
                    f"→ Linear(128→64) → ReLU → Dropout\n"
                    f"→ Linear(64→32) → ReLU\n"
                    f"→ Linear(32→{FORECAST_HORIZON})\n\n"
                    f"Total params: {ann_params:,}", language="text")
        with arch_col2:
            st.markdown("**RNN (Recurrent)**")
            rnn_params = sum(p.numel() for p in rnn_model.parameters())
            st.code(f"RNN(input={num_features}, hidden=64, layers=2, dropout=0.2)\n"
                    f"→ Take last hidden state\n"
                    f"→ Linear(64→{FORECAST_HORIZON})\n\n"
                    f"Total params: {rnn_params:,}", language="text")

        st.markdown("### 🏋️ Training ANN...")
        ann_history = train_model(ann_model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, LEARNING_RATE)
        st.success("ANN training complete!")

        st.markdown("### 🏋️ Training RNN...")
        rnn_history = train_model(rnn_model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, LEARNING_RATE)
        st.success("RNN training complete!")

        ann_metrics, ann_preds = evaluate_model(ann_model, X_test, y_test)
        rnn_metrics, rnn_preds = evaluate_model(rnn_model, X_test, y_test)

        st.session_state['trained'] = True
        st.session_state['ann_history'] = ann_history
        st.session_state['rnn_history'] = rnn_history
        st.session_state['ann_metrics'] = ann_metrics
        st.session_state['rnn_metrics'] = rnn_metrics
        st.session_state['ann_preds'] = ann_preds
        st.session_state['rnn_preds'] = rnn_preds
        st.session_state['y_test'] = y_test.numpy()
        st.session_state['ann_model'] = ann_model
        st.session_state['rnn_model'] = rnn_model
        st.session_state['scaler'] = scaler
        st.session_state['close_idx'] = close_idx
        st.session_state['num_features'] = num_features

    if st.session_state.get('trained'):
        ann_history = st.session_state['ann_history']
        rnn_history = st.session_state['rnn_history']
        ann_metrics = st.session_state['ann_metrics']
        rnn_metrics = st.session_state['rnn_metrics']
        ann_preds = st.session_state['ann_preds']
        rnn_preds = st.session_state['rnn_preds']
        y_test_np = st.session_state['y_test']

        st.markdown("### 📉 Loss Curves")
        epochs_range = list(range(1, len(ann_history['train_loss']) + 1))
        fig_loss = make_subplots(rows=1, cols=2, subplot_titles=('ANN', 'RNN'))
        fig_loss.add_trace(go.Scatter(x=epochs_range, y=ann_history['train_loss'], name='ANN Train', line=dict(color='#FF6B6B')), row=1, col=1)
        fig_loss.add_trace(go.Scatter(x=epochs_range, y=ann_history['val_loss'], name='ANN Val', line=dict(color='#FF6B6B', dash='dash')), row=1, col=1)
        fig_loss.add_trace(go.Scatter(x=epochs_range, y=rnn_history['train_loss'], name='RNN Train', line=dict(color='#4ECDC4')), row=1, col=2)
        fig_loss.add_trace(go.Scatter(x=epochs_range, y=rnn_history['val_loss'], name='RNN Val', line=dict(color='#4ECDC4', dash='dash')), row=1, col=2)
        fig_loss.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("### 📊 Model Comparison")
        results_df = pd.DataFrame([
            {'Model': 'ANN', **ann_metrics},
            {'Model': 'RNN', **rnn_metrics}
        ]).set_index('Model')
        st.dataframe(results_df, use_container_width=True)

        metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE (%)']
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ANN', x=metric_names, y=[ann_metrics[m] for m in metric_names], marker_color='#FF6B6B'))
        fig_bar.add_trace(go.Bar(name='RNN', x=metric_names, y=[rnn_metrics[m] for m in metric_names], marker_color='#4ECDC4'))
        fig_bar.update_layout(barmode='group', height=400, template='plotly_dark', title='Metrics Comparison (lower = better)')
        st.plotly_chart(fig_bar, use_container_width=True)

        winner = "ANN" if ann_metrics['RMSE'] < rnn_metrics['RMSE'] else "RNN"
        margin = abs(ann_metrics['RMSE'] - rnn_metrics['RMSE'])
        st.markdown(f'<div class="winner-banner">🏆 Winner: {winner} (RMSE margin: {margin:.6f})</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Predictions vs Actual")
        actuals_d1 = y_test_np[:, 0]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=actuals_d1, name='Actual', line=dict(color='white', width=2)))
        fig_pred.add_trace(go.Scatter(y=ann_preds[:, 0], name='ANN', line=dict(color='#FF6B6B', width=1.5)))
        fig_pred.add_trace(go.Scatter(y=rnn_preds[:, 0], name='RNN', line=dict(color='#4ECDC4', width=1.5)))
        fig_pred.update_layout(height=500, template='plotly_dark', title='Day 1 Forecast — Actual vs Predicted',
                               xaxis_title='Test Sample', yaxis_title='Scaled Close')
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("### 🎯 Scatter: Predicted vs Actual")
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=actuals_d1, y=ann_preds[:, 0], mode='markers', name='ANN',
                                    marker=dict(color='#FF6B6B', size=5, opacity=0.5)))
        fig_sc.add_trace(go.Scatter(x=actuals_d1, y=rnn_preds[:, 0], mode='markers', name='RNN',
                                    marker=dict(color='#4ECDC4', size=5, opacity=0.5)))
        mn, mx = actuals_d1.min(), actuals_d1.max()
        fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Perfect',
                                    line=dict(color='yellow', dash='dash', width=2)))
        fig_sc.update_layout(height=500, template='plotly_dark', title='Closer to yellow line = better',
                             xaxis_title='Actual', yaxis_title='Predicted')
        st.plotly_chart(fig_sc, use_container_width=True)

        pred_df = pd.DataFrame({'Actual': actuals_d1, 'ANN_Pred': ann_preds[:, 0], 'RNN_Pred': rnn_preds[:, 0]})
        csv_pred = pred_df.to_csv(index=False)
        st.download_button("📥 Download Predictions (CSV)", csv_pred, f"{TICKER}_predictions.csv", "text/csv")

    else:
        st.info("👈 Configure settings in the sidebar and click **🚀 Train Models** to begin.")


# ============================================================
# TAB 4: FORECAST
# ============================================================

with tab4:
    st.markdown("## 🔮 Future Forecast")

    if st.session_state.get('trained'):
        ann_model = st.session_state['ann_model']
        rnn_model = st.session_state['rnn_model']
        scaler = st.session_state['scaler']
        close_idx = st.session_state['close_idx']
        num_features = st.session_state['num_features']

        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                           'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                           'MACD', 'MACD_Signal', 'MACD_Histogram',
                           'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                           'ATR', 'Daily_Return', 'Log_Return']
        last_window = df[feature_columns].values[-LOOKBACK:]
        last_window_scaled = scaler.transform(last_window)
        input_tensor = torch.FloatTensor(last_window_scaled).unsqueeze(0)

        ann_model.eval()
        rnn_model.eval()
        with torch.no_grad():
            ann_forecast = ann_model(input_tensor).numpy()[0]
            rnn_forecast = rnn_model(input_tensor).numpy()[0]

        last_date = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON)

        fig_fc = go.Figure()
        hist_dates = df.index[-30:]
        hist_close_scaled = scaler.transform(df[feature_columns].values[-30:])[:, close_idx]
        fig_fc.add_trace(go.Scatter(x=hist_dates, y=hist_close_scaled, name='Historical',
                                    line=dict(color='white', width=2)))
        fig_fc.add_trace(go.Scatter(x=forecast_dates, y=ann_forecast, name='ANN Forecast',
                                    line=dict(color='#FF6B6B', width=2, dash='dot'),
                                    mode='lines+markers'))
        fig_fc.add_trace(go.Scatter(x=forecast_dates, y=rnn_forecast, name='RNN Forecast',
                                    line=dict(color='#4ECDC4', width=2, dash='dot'),
                                    mode='lines+markers'))
        fig_fc.update_layout(height=500, template='plotly_dark',
                             title=f'{TICKER} — Next {FORECAST_HORIZON} Business Days Forecast')
        st.plotly_chart(fig_fc, use_container_width=True)

        fc_df = pd.DataFrame({
            'Date': forecast_dates.strftime('%Y-%m-%d'),
            'ANN_Forecast': ann_forecast,
            'RNN_Forecast': rnn_forecast
        })
        st.dataframe(fc_df, use_container_width=True)

        csv_fc = fc_df.to_csv(index=False)
        st.download_button("📥 Download Forecast (CSV)", csv_fc, f"{TICKER}_forecast.csv", "text/csv")

        st.warning("⚠️ **Disclaimer:** This is for educational purposes only. Do NOT make financial decisions based on these predictions.")
    else:
        st.info("👈 Train models first in the **Training & Predictions** tab.")

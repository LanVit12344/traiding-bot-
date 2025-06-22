import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import time
import humanize
import sys
import os

# Add the project's root directory to the Python path.
# This is necessary for Streamlit to find the 'src' module when the script is run.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set page config and dark theme for plots
st.set_page_config(layout="wide", page_title="Binance Trading Bot")
pio.templates.default = "plotly_dark"

# --- Bot Components ---
from src.config import Config
from src.data_manager import DataManager
from src.indicator_engine import IndicatorEngine
from src.ai_model import AIModel
from src.news_sentiment import NewsSentimentAnalyzer
from src.strategy import Strategy
from src.risk_manager import RiskManager
from src.telegram_notifier import TelegramNotifier
from src.trader import Trader
from src.backtester import Backtester
from src.utils import initialize_logging

# --- Initialization ---
@st.cache_resource
def initialize_bot():
    """Initialize and cache all bot components."""
    try:
        config = Config("config.yaml")
        initialize_logging(config)
        data_manager = DataManager(config)
        indicator_engine = IndicatorEngine(config)
        ai_model = AIModel(config)
        news_analyzer = NewsSentimentAnalyzer(config)
        strategy = Strategy(config, indicator_engine, ai_model, news_analyzer)
        risk_manager = RiskManager(config)
        telegram_notifier = TelegramNotifier(config)
        trader = Trader(config, data_manager, strategy, risk_manager, telegram_notifier)
        backtester = Backtester(config, data_manager, indicator_engine, ai_model,
                                news_analyzer, strategy, risk_manager)
        return config, trader, data_manager, backtester
    except Exception as e:
        st.error(f"Fatal Error initializing bot: {e}")
        st.exception(e)
        st.stop()

config, trader, data_manager, backtester = initialize_bot()

# --- Helper Functions ---
def format_currency(value):
    return f"${value:,.2f}"

def format_delta(delta):
    return "normal" if delta >= 0 else "inverse"

# --- Main Logic ---
# Run one iteration of the bot logic to update the state before drawing the UI.
# Streamlit reruns the entire script on each interaction or refresh.
trader.run_once()

# --- UI Layout ---
st.title(f"🤖 Binance Trading Bot")
st.caption(f"Trading {config.get('trading.symbol')} on the {config.get('trading.timeframe')} timeframe.")

# Main Tabs
tab_live, tab_performance, tab_backtesting, tab_settings = st.tabs([
    "🔴 Live Analysis", 
    "📈 Performance & Trades",
    "🔬 Backtesting", 
    "⚙️ Settings"
])

# --- Live Analysis Tab ---
with tab_live:
    st.subheader("Real-Time Market Analysis")
    
    # Symbol selection dropdown
    available_symbols = config.get('trading.available_symbols', ['BTCUSDT'])
    if trader.symbol not in available_symbols:
        st.warning(f"Symbol {trader.symbol} not in available list. Defaulting to {available_symbols[0]}")
        trader.set_symbol(available_symbols[0])

    selected_symbol = st.selectbox(
        "Select Trading Symbol",
        options=available_symbols,
        index=available_symbols.index(trader.symbol)
    )

    if selected_symbol != trader.symbol:
        trader.set_symbol(selected_symbol)
        st.success(f"Symbol changed to {selected_symbol}. Data is being updated.")
        st.rerun()

    # Placeholder for the main chart and analysis
    chart_placeholder = st.empty()
    
    # Fetch data and display loading indicator
    with st.spinner("Fetching latest market data..."):
        market_data = trader.get_market_data()

    if market_data is None or market_data.empty or len(market_data) < 20:
         with chart_placeholder.container():
            st.warning("Could not fetch sufficient market data to display chart. Please try again later.")
    else:
        with chart_placeholder.container():
            st.plotly_chart(trader.get_analysis_chart(), use_container_width=True)

    # Analysis columns
    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.metric("Current Price", format_currency(market_data.iloc[-1]['close']) if not market_data.empty else "N/A")
    with cols[1]:
        st.metric("24h Change", f"{market_data['close'].pct_change(periods=24).iloc[-1]:.2%}" if not market_data.empty and len(market_data) > 24 else "N/A")
    
    # Display latest analysis from the trader state
    latest_analysis = trader.last_analysis
    if latest_analysis:
        rec = latest_analysis.get('final_recommendation', 'hold').upper()
        color = "green" if rec == "BUY" else "red" if rec == "SELL" else "orange"
        cols[2].markdown(f"**Recommendation:** <span style='color:{color}; font-size: 1.2em;'>{rec}</span>", unsafe_allow_html=True)

    if latest_analysis:
        st.subheader("Decision Breakdown")
        analysis = latest_analysis
        analysis_cols = st.columns(3)
        with analysis_cols[0]:
            st.markdown("##### Technical Indicators")
            st.json(analysis.get('indicator_signals', {'error': 'Not available'}))
        with analysis_cols[1]:
            st.markdown("##### AI Model Prediction")
            st.json(analysis.get('ai_signal', {'error': 'Not available'}))
        with analysis_cols[2]:
            st.markdown("##### News Sentiment")
            st.json(analysis.get('sentiment_signal', {'error': 'Not available'}))

# --- Performance & Trades Tab ---
with tab_performance:
    st.subheader("Trade History & Equity")
    
    status = trader.get_status()
    trade_history = status.get('trade_history', [])
    
    if trade_history:
        df_trades = pd.DataFrame(trade_history)
        total_pnl = df_trades['pnl'].sum()
        num_trades = len(df_trades)
        win_rate = (df_trades['pnl'] > 0).sum() / num_trades if num_trades > 0 else 0
        
        perf_cols = st.columns(3)
        perf_cols[0].metric("Total PNL", format_currency(total_pnl), delta_color=format_delta(total_pnl))
        perf_cols[1].metric("Total Trades", num_trades)
        perf_cols[2].metric("Win Rate", f"{win_rate:.2%}")
        
        initial_balance = config.get('backtest.initial_balance')
        df_trades['equity'] = initial_balance + df_trades['pnl'].cumsum()
        
        fig = px.line(df_trades, x='exit_time', y='equity', title='Live/Paper Trading Equity Curve', markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Equity ($)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades have been executed yet.")

    st.subheader("All Executed Trades")
    if trade_history:
        st.dataframe(pd.DataFrame(trade_history), use_container_width=True)
    else:
        st.info("The trade history is empty.")

# --- Backtesting Tab ---
with tab_backtesting:
    st.subheader("🧪 Strategy Backtesting")
    
    st.info("Test your strategy settings against historical data.")

    backtest_cols = st.columns([2, 1, 1, 1])
    with backtest_cols[0]:
        bt_symbol = st.selectbox("Symbol", config.get('trading.available_symbols'), 
                                 index=config.get('trading.available_symbols').index(config.get('trading.symbol')), 
                                 key="bt_symbol")
    with backtest_cols[1]:
        default_start = datetime.now() - pd.DateOffset(months=6)
        bt_start_date = st.date_input("Start Date", value=default_start)
    with backtest_cols[2]:
        bt_end_date = st.date_input("End Date", value=datetime.now())
    with backtest_cols[3]:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

    if run_button:
        with st.spinner(f"Running backtest for {bt_symbol} from {bt_start_date} to {bt_end_date}..."):
            try:
                # Make sure the symbol in the config is updated for the backtester
                config.set('trading.symbol', bt_symbol)
                results = backtester.run(
                    start_date=bt_start_date.strftime('%Y-%m-%d'),
                    end_date=bt_end_date.strftime('%Y-%m-%d')
                )
                st.session_state.backtest_results = results
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)
                st.session_state.backtest_results = None

    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        st.success("Backtest complete!")
        results = st.session_state.backtest_results
        
        st.markdown("---")
        bt_metric_cols = st.columns(5)
        bt_metric_cols[0].metric("Total Return", f"{results.get('total_return', 0):.2%}", delta_color=format_delta(results.get('total_return', 0)))
        bt_metric_cols[1].metric("Total Trades", results.get('total_trades', 0))
        bt_metric_cols[2].metric("Win Rate", f"{results.get('win_rate', 0):.2%}")
        bt_metric_cols[3].metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
        bt_metric_cols[4].metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        
        df_equity = results.get('equity_curve')
        if df_equity is not None and not df_equity.empty:
            fig_equity = px.line(df_equity, y='equity', title=f'Equity Curve for {bt_symbol}')
            st.plotly_chart(fig_equity, use_container_width=True)
            
        with st.expander("View Backtest Trade History"):
            df_trades = results.get('trades')
            if df_trades is not None and not df_trades.empty:
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.warning("No trades were executed in this backtest.")

# --- Settings Tab ---
with tab_settings:
    st.subheader("🔧 Bot Configuration")
    
    st.warning("Changing settings will restart the bot. Unsaved analysis will be lost.")
    
    with st.expander("Risk Management Settings", expanded=True):
        risk_config = config.get('risk_management')
        use_trailing_stop = st.checkbox("Use Trailing Stop", value=risk_config.get('use_trailing_stop'))
        stop_loss_mult = st.number_input("Stop Loss ATR Multiplier", value=risk_config.get('stop_loss.atr_multiplier', 2.0))
        take_profit_mult = st.number_input("Take Profit ATR Multiplier", value=risk_config.get('take_profit.atr_multiplier', 3.0))

    with st.expander("AI & Strategy Settings"):
         ai_config = config.get('ai')
         confidence_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, value=ai_config.get('confidence_threshold', 0.6))
    
    if st.button("Save Settings & Restart Bot"):
        config.set('risk_management.use_trailing_stop', use_trailing_stop)
        config.set('risk_management.stop_loss.atr_multiplier', stop_loss_mult)
        config.set('risk_management.take_profit.atr_multiplier', take_profit_mult)
        config.set('ai.confidence_threshold', confidence_threshold)
        
        st.cache_resource.clear()
        st.success("Settings saved! Restarting bot...")
        time.sleep(2)
        st.rerun()

# --- Auto-refresh logic ---
# This part runs at the end of the script. 
# It pauses for the configured interval and then tells Streamlit to rerun the script from the top.
# This is the correct way to create a "live" dashboard in Streamlit.
try:
    refresh_interval = config.get('dashboard.refresh_interval', 10)
    time.sleep(refresh_interval)
    st.rerun()
except Exception as e:
    # This might happen if the user closes the window while the script is sleeping.
    st.error(f"An error occurred during the refresh loop: {e}")
"""
Data manager for fetching market data from Binance.
"""
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from binance import Client, AsyncClient
from binance.exceptions import BinanceAPIException
import websocket
import threading
import time

class DataManager:
    """Manages real-time and historical market data from Binance."""
    
    def __init__(self, config):
        """Initialize data manager with configuration."""
        self.config = config
        self.client = None
        self.ws_manager = None
        self.ws_connection = None
        self.callbacks = []
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        self._init_client()
    
    def _init_client(self):
        """Initialize Binance client with API credentials."""
        try:
            api_key = self.config.get('binance.api_key')
            secret_key = self.config.get('binance.secret_key')
            
            if not api_key or not secret_key:
                self.logger.warning("Binance API credentials not found. Using public client.")
                self.client = Client()
            else:
                self.client = Client(api_key, secret_key)
                self.logger.info("Binance client initialized with API credentials")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def get_historical_data(self, symbol: str, interval: str, 
                          start_time: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start time in string format
            end_date: End time in string format
            limit: Maximum number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            klines = self.client.get_historical_klines(
                symbol, interval, start_str=start_time, end_str=end_date, limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            raise
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            if self.config.is_live_mode():
                return self.client.get_account()
            else:
                # Return mock data for paper trading
                return {
                    'balances': [
                        {'asset': 'USDT', 'free': '10000', 'locked': '0'},
                        {'asset': 'BTC', 'free': '0', 'locked': '0'}
                    ]
                }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information including filters."""
        try:
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
            raise ValueError(f"Symbol {symbol} not found")
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            raise
    
    def start_websocket(self, symbol: str, callback: Callable):
        """
        Start WebSocket connection for real-time data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'btcusdt')
            callback: Function to call when new data arrives
        """
        try:
            # Stop any existing websocket
            if self.is_connected:
                self.stop_websocket()
                time.sleep(1) # Give it a moment to close

            if not self.callbacks:
                self.callbacks.append(callback)
            
            if not self.is_connected:
                # Create WebSocket connection using websocket-client
                stream_name = f"{symbol.lower()}@kline_{self.config.get('trading.timeframe')}"
                ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
                
                self.ws_connection = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._handle_websocket_message,
                    on_error=self._handle_websocket_error,
                    on_close=self._handle_websocket_close,
                    on_open=self._handle_websocket_open
                )
                
                # Start WebSocket in a separate thread
                self.ws_thread = threading.Thread(target=self.ws_connection.run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                self.is_connected = True
                self.logger.info(f"WebSocket started for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error starting WebSocket: {e}")
            raise
    
    def _handle_websocket_open(self, ws):
        """Handle WebSocket connection open."""
        self.logger.info("WebSocket connection opened")
    
    def _handle_websocket_error(self, ws, error):
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")
    
    def _handle_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.logger.info("WebSocket connection closed")
        self.is_connected = False
    
    def _handle_websocket_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            if 'k' in data:  # Kline/Candlestick data
                kline = data['k']
                
                # Convert to DataFrame format
                candle_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']
                }
                
                # Call all registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(candle_data)
                    except Exception as e:
                        self.logger.error(f"Error in WebSocket callback: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    def stop_websocket(self):
        """Stop WebSocket connection."""
        try:
            if self.ws_connection:
                self.ws_connection.close()
                self.is_connected = False
                self.logger.info("WebSocket stopped")
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol."""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol."""
        try:
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return order_book
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            raise
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24-hour ticker statistics."""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Error getting 24hr ticker: {e}")
            raise 
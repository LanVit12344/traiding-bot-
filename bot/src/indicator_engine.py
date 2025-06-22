"""
Technical indicators engine for calculating various market indicators.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple, Optional
import logging

class IndicatorEngine:
    """Engine for calculating technical indicators."""
    
    def __init__(self, config):
        """Initialize indicator engine with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        try:
            df = df.copy()
            
            # Calculate RSI
            df = self.add_rsi(df)
            
            # Calculate MACD
            df = self.add_macd(df)
            
            # Calculate Bollinger Bands
            df = self.add_bollinger_bands(df)
            
            # Calculate EMAs
            df = self.add_ema(df)
            
            # Calculate ATR
            df = self.add_atr(df)
            
            # Calculate additional indicators
            df = self.add_stochastic(df)
            df = self.add_williams_r(df)
            df = self.add_cci(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise
    
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator to DataFrame."""
        try:
            period = self.config.get('indicators.rsi.period', 14)
            df['rsi'] = ta.momentum.RSIIndicator(
                close=df['close'], 
                window=period
            ).rsi()
            return df
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator to DataFrame."""
        try:
            fast_period = self.config.get('indicators.macd.fast_period', 12)
            slow_period = self.config.get('indicators.macd.slow_period', 26)
            signal_period = self.config.get('indicators.macd.signal_period', 9)
            
            macd = ta.trend.MACD(
                close=df['close'],
                window_fast=fast_period,
                window_slow=slow_period,
                window_sign=signal_period
            )
            
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands indicator to DataFrame."""
        try:
            period = self.config.get('indicators.bollinger_bands.period', 20)
            std_dev = self.config.get('indicators.bollinger_bands.std_dev', 2)
            
            bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=period,
                window_dev=std_dev
            )
            
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_percent'] = bb.bollinger_pband()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages to DataFrame."""
        try:
            short_period = self.config.get('indicators.ema.short_period', 12)
            long_period = self.config.get('indicators.ema.long_period', 26)
            trend_period = self.config.get('indicators.ema.trend_period', 200)
            
            df['ema_short'] = ta.trend.EMAIndicator(
                close=df['close'], 
                window=short_period
            ).ema_indicator()
            
            df['ema_long'] = ta.trend.EMAIndicator(
                close=df['close'], 
                window=long_period
            ).ema_indicator()
            
            df['ema_trend'] = ta.trend.EMAIndicator(
                close=df['close'], 
                window=trend_period
            ).ema_indicator()
            
            # EMA crossover signals
            df['ema_cross'] = np.where(
                df['ema_short'] > df['ema_long'], 1, -1
            )
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating EMAs: {e}")
            return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range indicator to DataFrame."""
        try:
            period = self.config.get('indicators.atr.period', 14)
            
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            ).average_true_range()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df
    
    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator to DataFrame."""
        try:
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return df
    
    def add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Williams %R indicator to DataFrame."""
        try:
            df['williams_r'] = ta.momentum.WilliamsRIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            ).williams_r()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return df
    
    def add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Commodity Channel Index to DataFrame."""
        try:
            df['cci'] = ta.trend.CCIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            ).cci()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return df
    
    def get_support_resistance_levels(self, df: pd.DataFrame, 
                                    window: int = 20) -> Tuple[float, float]:
        """
        Calculate support and resistance levels using recent price action.
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for calculations
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        try:
            recent_data = df.tail(window)
            
            # Support level (recent low)
            support_level = recent_data['low'].min()
            
            # Resistance level (recent high)
            resistance_level = recent_data['high'].max()
            
            return support_level, resistance_level
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return None, None
    
    def get_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate price volatility using standard deviation of returns.
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for calculations
            
        Returns:
            Volatility value
        """
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(window).std()
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def get_trend_strength(self, df: pd.DataFrame) -> str:
        """
        Determine trend strength based on multiple indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Trend strength: 'strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish'
        """
        try:
            if df.empty or len(df) < 50:
                return 'neutral'
            
            latest = df.iloc[-1]
            
            # Check EMA trend
            ema_trend = latest['close'] > latest['ema_trend']
            
            # Check RSI
            rsi_bullish = latest['rsi'] > 50
            rsi_oversold = latest['rsi'] < 30
            rsi_overbought = latest['rsi'] > 70
            
            # Check MACD
            macd_bullish = latest['macd'] > latest['macd_signal']
            
            # Check Bollinger Bands position
            bb_position = latest['bb_percent']
            
            # Determine trend strength
            bullish_signals = sum([
                ema_trend,
                rsi_bullish,
                macd_bullish,
                bb_position > 0.5
            ])
            
            bearish_signals = sum([
                not ema_trend,
                not rsi_bullish,
                not macd_bullish,
                bb_position < 0.5
            ])
            
            if bullish_signals >= 3:
                return 'strong_bullish' if rsi_overbought else 'bullish'
            elif bearish_signals >= 3:
                return 'strong_bearish' if rsi_oversold else 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error determining trend strength: {e}")
            return 'neutral'
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate overall signal strength based on multiple indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Signal strength between -1 (strong sell) and 1 (strong buy)
        """
        try:
            if df.empty or len(df) < 50:
                return 0.0
            
            latest = df.iloc[-1]
            
            # RSI signal (-1 to 1)
            rsi_signal = 0
            if latest['rsi'] < 30:
                rsi_signal = 1  # Oversold - buy signal
            elif latest['rsi'] > 70:
                rsi_signal = -1  # Overbought - sell signal
            else:
                rsi_signal = (latest['rsi'] - 50) / 50  # Normalized to -1 to 1
            
            # MACD signal (-1 to 1)
            macd_signal = 0
            if latest['macd'] > latest['macd_signal']:
                macd_signal = min(1, latest['macd_histogram'] / latest['close'] * 100)
            else:
                macd_signal = max(-1, latest['macd_histogram'] / latest['close'] * 100)
            
            # EMA signal (-1 to 1)
            ema_signal = 1 if latest['ema_cross'] > 0 else -1
            
            # Bollinger Bands signal (-1 to 1)
            bb_signal = (latest['bb_percent'] - 0.5) * 2  # Convert 0-1 to -1 to 1
            
            # Weighted average of signals
            weights = [0.3, 0.3, 0.2, 0.2]  # RSI, MACD, EMA, BB
            signals = [rsi_signal, macd_signal, ema_signal, bb_signal]
            
            weighted_signal = sum(w * s for w, s in zip(weights, signals))
            
            return max(-1, min(1, weighted_signal))  # Clamp between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0 
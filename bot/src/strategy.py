"""
Trading strategy module that combines technical indicators, AI signals, and sentiment.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class Strategy:
    """Main trading strategy that combines all signals."""
    
    def __init__(self, config, indicator_engine, ai_model, news_analyzer):
        """Initialize strategy with all components."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicator_engine = indicator_engine
        self.ai_model = ai_model
        self.news_analyzer = news_analyzer
        
        # Strategy parameters
        self.rsi_oversold = config.get('indicators.rsi.oversold', 30)
        self.rsi_overbought = config.get('indicators.rsi.overbought', 70)
        self.min_signal_strength = 0.6
        self.min_ai_confidence = 0.7
        
        # Signal weights
        self.weights = {
            'technical': 0.4,
            'ai': 0.4,
            'sentiment': 0.2
        }
    
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive market analysis.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if df.empty or len(df) < 50:
                return self._get_neutral_analysis()
            
            # Calculate indicators if not present
            if 'rsi' not in df.columns:
                df = self.indicator_engine.calculate_all_indicators(df)
            
            latest = df.iloc[-1]
            
            # Get signals from all sources. Ensure they always return a dict.
            indicator_signals = self._get_technical_signal(df)
            ai_signal = self._get_ai_signal(df)
            sentiment_signal = self._get_sentiment_signal()
            
            # Combine signals
            combined_signal = self._combine_signals(indicator_signals, ai_signal, sentiment_signal)
            
            # Get trend analysis
            trend_strength = self.indicator_engine.get_trend_strength(df)
            signal_strength = self.indicator_engine.get_signal_strength(df)
            
            # Get support/resistance levels
            support, resistance = self.indicator_engine.get_support_resistance_levels(df)
            
            # Get volatility
            volatility = self.indicator_engine.get_volatility(df)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'current_price': latest['close'],
                'indicator_signals': indicator_signals,
                'ai_signal': ai_signal,
                'sentiment_signal': sentiment_signal,
                'combined_signal': combined_signal,
                'trend_strength': trend_strength,
                'signal_strength': signal_strength,
                'support_level': support,
                'resistance_level': resistance,
                'volatility': volatility,
                'indicators': {
                    'rsi': latest.get('rsi', 50),
                    'macd': latest.get('macd', 0),
                    'macd_signal': latest.get('macd_signal', 0),
                    'bb_percent': latest.get('bb_percent', 0.5),
                    'ema_trend': latest.get('ema_trend', latest['close']),
                    'atr': latest.get('atr', 0)
                },
                'final_recommendation': self._get_recommendation(combined_signal),
                'confidence': self._calculate_confidence(indicator_signals, ai_signal, sentiment_signal)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {e}")
            return self._get_neutral_analysis()
    
    def _get_technical_signal(self, df: pd.DataFrame) -> Dict:
        """Get technical analysis signal."""
        try:
            if df.empty or len(df) < 50:
                return {'signal': 'hold', 'strength': 0.0, 'reasons': ['Insufficient data']}
            
            latest = df.iloc[-1]
            reasons = []
            signal_strength = 0.0
            
            # RSI analysis
            rsi = latest.get('rsi', 50)
            if rsi < self.rsi_oversold:
                signal_strength += 0.3
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.rsi_overbought:
                signal_strength -= 0.3
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD analysis
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal:
                signal_strength += 0.2
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal:
                signal_strength -= 0.2
                reasons.append("MACD bearish crossover")
            
            # Bollinger Bands analysis
            bb_percent = latest.get('bb_percent', 0.5)
            if bb_percent < 0.2:
                signal_strength += 0.2
                reasons.append("Price near lower Bollinger Band")
            elif bb_percent > 0.8:
                signal_strength -= 0.2
                reasons.append("Price near upper Bollinger Band")
            
            # EMA trend analysis
            ema_trend = latest.get('ema_trend', latest['close'])
            if latest['close'] > ema_trend:
                signal_strength += 0.1
                reasons.append("Price above EMA trend")
            else:
                signal_strength -= 0.1
                reasons.append("Price below EMA trend")
            
            # Determine signal
            if signal_strength > 0.3:
                signal = 'buy'
            elif signal_strength < -0.3:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'signal': signal,
                'strength': abs(signal_strength),
                'reasons': reasons
            }
            
        except Exception as e:
            self.logger.error(f"Error getting technical signal: {e}")
            return {'signal': 'hold', 'strength': 0.0, 'reasons': ['Technical analysis error']}
    
    def _get_ai_signal(self, df: pd.DataFrame) -> Dict:
        """Get AI model signal."""
        try:
            if self.ai_model is None:
                return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['AI model not available']}
            
            signal, confidence = self.ai_model.get_signal(df)
            
            reasons = []
            if confidence > self.min_ai_confidence:
                reasons.append(f"AI model confident ({confidence:.2f})")
            else:
                reasons.append(f"AI model low confidence ({confidence:.2f})")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasons': reasons
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI signal: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['AI analysis error']}
    
    def _get_sentiment_signal(self) -> Dict:
        """Get sentiment analysis signal."""
        try:
            if self.news_analyzer is None:
                return {'signal': 'hold', 'strength': 0.0, 'reasons': ['Sentiment analysis not available']}
            
            sentiment_score, description = self.news_analyzer.get_overall_sentiment()
            
            reasons = [f"Market sentiment: {description}"]
            
            # Convert sentiment to signal
            if sentiment_score > 0.3:
                signal = 'buy'
                strength = sentiment_score
            elif sentiment_score < -0.3:
                signal = 'sell'
                strength = abs(sentiment_score)
            else:
                signal = 'hold'
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': strength,
                'reasons': reasons
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment signal: {e}")
            return {'signal': 'hold', 'strength': 0.0, 'reasons': ['Sentiment analysis error']}
    
    def _combine_signals(self, technical: Dict, ai: Dict, sentiment: Dict) -> Dict:
        """Combine all signals into a final decision."""
        try:
            # Convert signals to numerical values
            signal_values = {
                'buy': 1,
                'hold': 0,
                'sell': -1
            }
            
            # Get signal values
            tech_value = signal_values.get(technical['signal'], 0) * technical['strength']
            ai_value = signal_values.get(ai['signal'], 0) * ai['confidence']
            sent_value = signal_values.get(sentiment['signal'], 0) * sentiment['strength']
            
            # Weighted combination
            combined_value = (
                tech_value * self.weights['technical'] +
                ai_value * self.weights['ai'] +
                sent_value * self.weights['sentiment']
            )
            
            # Determine final signal
            if combined_value > self.min_signal_strength:
                signal = 'buy'
            elif combined_value < -self.min_signal_strength:
                signal = 'sell'
            else:
                signal = 'hold'
            
            # Collect all reasons
            all_reasons = []
            all_reasons.extend(technical.get('reasons', []))
            all_reasons.extend(ai.get('reasons', []))
            all_reasons.extend(sentiment.get('reasons', []))
            
            return {
                'signal': signal,
                'strength': abs(combined_value),
                'value': combined_value,
                'reasons': all_reasons
            }
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return {'signal': 'hold', 'strength': 0.0, 'value': 0.0, 'reasons': ['Signal combination error']}
    
    def _get_recommendation(self, combined_signal: Dict) -> str:
        """Get trading recommendation based on combined signal."""
        try:
            signal = combined_signal['signal']
            strength = combined_signal['strength']
            
            if signal == 'buy' and strength > 0.8:
                return 'strong_buy'
            elif signal == 'buy' and strength > 0.6:
                return 'buy'
            elif signal == 'sell' and strength > 0.8:
                return 'strong_sell'
            elif signal == 'sell' and strength > 0.6:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            self.logger.error(f"Error getting recommendation: {e}")
            return 'hold'
    
    def _calculate_confidence(self, technical: Dict, ai: Dict, sentiment: Dict) -> float:
        """Calculate overall confidence in the analysis."""
        try:
            # Weighted average of individual confidences
            tech_confidence = technical.get('strength', 0.0)
            ai_confidence = ai.get('confidence', 0.0)
            sent_confidence = sentiment.get('strength', 0.0)
            
            weighted_confidence = (
                tech_confidence * self.weights['technical'] +
                ai_confidence * self.weights['ai'] +
                sent_confidence * self.weights['sentiment']
            )
            
            return min(1.0, weighted_confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def should_enter_position(self, analysis: Dict, current_positions: int) -> Tuple[bool, Dict]:
        """
        Determine if a new position should be entered.
        
        Args:
            analysis: Market analysis results
            current_positions: Number of current open positions
            
        Returns:
            Tuple of (should_enter, entry_details)
        """
        try:
            recommendation = analysis['final_recommendation']
            confidence = analysis['confidence']
            
            # Check if we have a strong enough signal
            if recommendation not in ['strong_buy', 'buy']:
                return False, {'reason': 'No buy signal'}
            
            if confidence < 0.6:
                return False, {'reason': 'Low confidence'}
            
            # Check position limits
            max_positions = self.config.get('trading.max_positions', 3)
            if current_positions >= max_positions:
                return False, {'reason': 'Maximum positions reached'}
            
            # Check trend strength
            trend_strength = analysis['trend_strength']
            if trend_strength in ['strong_bearish', 'bearish']:
                return False, {'reason': 'Bearish trend'}
            
            # Entry details
            entry_details = {
                'signal': analysis['combined_signal']['signal'],
                'confidence': confidence,
                'reasons': analysis['combined_signal']['reasons'],
                'current_price': analysis['current_price'],
                'support_level': analysis['support_level'],
                'resistance_level': analysis['resistance_level'],
                'atr': analysis['indicators']['atr']
            }
            
            return True, entry_details
            
        except Exception as e:
            self.logger.error(f"Error checking entry conditions: {e}")
            return False, {'reason': 'Error in entry analysis'}
    
    def should_exit_position(self, analysis: Dict, position: Dict) -> Tuple[bool, Dict]:
        """
        Determine if an existing position should be exited.
        
        Args:
            analysis: Market analysis results
            position: Current position details
            
        Returns:
            Tuple of (should_exit, exit_details)
        """
        try:
            recommendation = analysis['final_recommendation']
            confidence = analysis['confidence']
            
            # Check for strong exit signals
            if recommendation in ['strong_sell', 'sell']:
                return True, {
                    'reason': 'Sell signal',
                    'type': 'signal_exit',
                    'confidence': confidence
                }
            
            # Check for trend reversal
            trend_strength = analysis['trend_strength']
            position_direction = position.get('direction', 'long')
            
            if position_direction == 'long' and trend_strength in ['strong_bearish', 'bearish']:
                return True, {
                    'reason': 'Trend reversal',
                    'type': 'trend_exit',
                    'confidence': confidence
                }
            
            if position_direction == 'short' and trend_strength in ['strong_bullish', 'bullish']:
                return True, {
                    'reason': 'Trend reversal',
                    'type': 'trend_exit',
                    'confidence': confidence
                }
            
            # Check for technical exit conditions
            rsi = analysis['indicators']['rsi']
            if position_direction == 'long' and rsi > self.rsi_overbought:
                return True, {
                    'reason': 'RSI overbought',
                    'type': 'technical_exit',
                    'confidence': confidence
                }
            
            if position_direction == 'short' and rsi < self.rsi_oversold:
                return True, {
                    'reason': 'RSI oversold',
                    'type': 'technical_exit',
                    'confidence': confidence
                }
            
            return False, {'reason': 'No exit signal'}
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False, {'reason': 'Error in exit analysis'}
    
    def _get_neutral_analysis(self) -> Dict:
        """Return a neutral/default analysis structure."""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': None,
            'indicator_signals': {'signal': 'hold', 'strength': 0.0, 'reasons': ['No data']},
            'ai_signal': {'signal': 'hold', 'confidence': 0.0, 'reasons': ['No data']},
            'sentiment_signal': {'signal': 'hold', 'strength': 0.0, 'reasons': ['No data']},
            'combined_signal': {'signal': 'hold', 'strength': 0.0},
            'trend_strength': {'trend': 'sideways', 'strength': 0.0},
            'signal_strength': 0.0,
            'support_level': None,
            'resistance_level': None,
            'volatility': 0.0,
            'indicators': {},
            'final_recommendation': 'Wait',
            'confidence': 0.0
        } 
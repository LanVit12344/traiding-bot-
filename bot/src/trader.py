"""
Main trading engine that coordinates all components.
"""
import pandas as pd
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import threading
from binance.exceptions import BinanceAPIException

class Trader:
    """Main trading engine that coordinates all components."""
    
    def __init__(self, config, data_manager, strategy, risk_manager, telegram_notifier):
        """Initialize trader with all components."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.telegram_notifier = telegram_notifier
        
        # Trading state
        self.is_running = False
        self.positions = []
        self.trade_history = []
        self.account_balance = 10000  # Default for paper trading
        self.last_analysis = None
        self._data_cache = None
        self._cache_timestamp = None
        
        # Trading parameters
        self.symbol = config.get('trading.symbol', 'BTCUSDT')
        self.timeframe = config.get('trading.timeframe', '1h')
        self.mode = config.get('trading.mode', 'paper')
        
        # Performance tracking
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            log_config = self.config.get_logging_config()
            log_file = log_config.get('file', 'trades.log')
            log_level = getattr(logging, log_config.get('level', 'INFO'))
            
            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            self.logger.setLevel(log_level)
            
        except Exception as e:
            self.logger.error(f"Error setting up logging: {e}")
    
    def start(self):
        """Start the trading bot."""
        try:
            self.logger.info("Starting trading bot...")
            
            # Test connections
            if not self._test_connections():
                self.logger.error("Connection tests failed")
                return False
            
            # Initialize account
            self._initialize_account()
            
            # Start WebSocket for real-time data
            self.data_manager.start_websocket(self.symbol, self._on_data_update)
            
            self.is_running = True
            self.logger.info("Trading bot started successfully")
            
            # Send startup notification
            self.telegram_notifier.send_message(
                f"🤖 Trading bot started\n"
                f"Symbol: {self.symbol}\n"
                f"Mode: {self.mode}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            self.telegram_notifier.notify_error(str(e), "Bot startup")
            return False
    
    def stop(self):
        """Stop the trading bot."""
        try:
            self.logger.info("Stopping trading bot...")
            
            self.is_running = False
            
            # Close all positions
            self._close_all_positions()
            
            # Stop WebSocket
            self.data_manager.stop_websocket()
            
            # Send shutdown notification
            self.telegram_notifier.send_message(
                f"🛑 Trading bot stopped\n"
                f"Total P&L: ${self.total_pnl:,.2f}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.logger.info("Trading bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {e}")
    
    def _test_connections(self) -> bool:
        """Test all connections."""
        try:
            # Test Binance connection
            try:
                self.data_manager.get_current_price(self.symbol)
                self.logger.info("Binance connection: OK")
            except Exception as e:
                self.logger.error(f"Binance connection failed: {e}")
                return False
            
            # Test Telegram connection
            if self.telegram_notifier.enabled:
                if not self.telegram_notifier.test_connection():
                    self.logger.warning("Telegram connection failed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing connections: {e}")
            return False
    
    def _initialize_account(self):
        """Initialize account information."""
        try:
            if self.mode == 'live':
                account_info = self.data_manager.get_account_info()
                # Extract balance from account info
                for balance in account_info.get('balances', []):
                    if balance['asset'] == 'USDT':
                        self.account_balance = float(balance['free'])
                        break
            else:
                # Paper trading - use default balance
                self.account_balance = 10000
            
            self.logger.info(f"Account balance: ${self.account_balance:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error initializing account: {e}")
    
    def _on_data_update(self, data: Dict):
        """Handle real-time data updates."""
        try:
            if not self.is_running:
                return
            
            # Process the update in a separate thread to avoid blocking
            threading.Thread(target=self._process_data_update, args=(data,)).start()
            
        except Exception as e:
            self.logger.error(f"Error handling data update: {e}")
    
    def _process_data_update(self, data: Dict):
        """Process real-time data update."""
        try:
            # The 'k' key holds the kline data
            kline = data.get('k', {})
            close_price = float(kline.get('c'))
            
            self.logger.info(f"New data for {self.symbol}: Close price is {close_price:.2f}")
            # Get historical data for analysis
            df = self._get_analysis_data()
            
            if df is None or df.empty:
                self.logger.warning("Could not get analysis data. Skipping cycle.")
                return
            
            # Perform market analysis
            self.logger.info("Performing market analysis...")
            analysis = self.strategy.analyze_market(df)
            self.last_analysis = analysis
            self.logger.info(f"Analysis complete. Recommendation: {analysis.get('recommendation', 'hold').upper()}")
            
            # Check for entry signals
            self._check_entry_signals(analysis)
            
            # Check for exit signals
            self._check_exit_signals(analysis)
            
            # Update trailing stops
            self._update_trailing_stops(close_price)
            
        except Exception as e:
            self.logger.error(f"Error processing data update: {e}")
    
    def run_once(self):
        """
        Executes one full analysis and trading cycle.
        This is designed to be called repeatedly by an external scheduler (like Streamlit's loop).
        """
        try:
            if not self.is_running:
                # In paper/live mode, this would be controlled by start/stop buttons.
                # For dashboard purposes, we can consider it "running" for analysis.
                # self.logger.info("Trader is not running. Skipping cycle.")
                # For now, we let it run to populate the dashboard.
                pass

            df = self._get_analysis_data()
            if df is None or df.empty:
                self.logger.warning("Could not get analysis data for this cycle.")
                return

            # Perform market analysis
            analysis = self.strategy.analyze_market(df)
            self.last_analysis = analysis
            
            # The following checks would typically lead to trades.
            # In a non-threaded, dashboard-driven context, they update the state.
            self._check_entry_signals(analysis)
            self._check_exit_signals(analysis)
            
            # Update trailing stops if there are active positions
            if self.positions:
                current_price = df.iloc[-1]['close']
                self._update_trailing_stops(current_price)
            
            self.logger.debug("Trader run_once cycle complete.")

        except Exception as e:
            self.logger.error(f"Error during run_once cycle: {e}")
            self.telegram_notifier.notify_error(str(e), "run_once")
            
    def get_market_data(self) -> Optional[pd.DataFrame]:
        """Returns the latest market data used for analysis."""
        return self._get_analysis_data()

    def get_analysis_chart(self):
        """
        Generates a Plotly chart with price, indicators, and trades.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df = self.get_market_data()
        if df is None or df.empty:
            return go.Figure()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], 
                                     low=df['low'], close=df['close'], name='Price'), 
                      row=1, col=1)

        # Indicators on price chart
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_short'], name='EMA Short', line=dict(color='yellow', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_long'], name='EMA Long', line=dict(color='orange', width=1)), row=1, col=1)
        if 'bb_high' in df.columns and 'bb_low' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], name='BB High', line=dict(color='rgba(152,251,152,0.5)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], name='BB Low', line=dict(color='rgba(255,182,193,0.5)', width=1)), row=1, col=1)

        # RSI on second chart
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line_color='purple'), row=2, col=1)
            fig.add_hline(y=self.config.get('indicators.rsi.overbought', 70), line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=self.config.get('indicators.rsi.oversold', 30), line_dash="dash", line_color="green", row=2, col=1)

        # Layout settings
        fig.update_layout(
            title_text=f"{self.symbol} Market Analysis ({self.timeframe})",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        return fig
    
    def _get_analysis_data(self) -> Optional[pd.DataFrame]:
        """Get data for analysis, using a cache to avoid excessive API calls."""
        try:
            cache_duration = timedelta(seconds=60)  # Cache data for 60 seconds
            
            # Check if cache is valid
            if self._data_cache is not None and \
               self._cache_timestamp is not None and \
               datetime.now() - self._cache_timestamp < cache_duration:
                self.logger.debug("Returning cached analysis data.")
                return self._data_cache

            # Get historical data
            days_to_fetch = self.config.get('trading.data_fetch_days', 30)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_to_fetch)
            
            df = self.data_manager.get_historical_data(
                symbol=self.symbol,
                interval=self.timeframe,
                start_time=start_time.strftime('%Y-%m-%d'),
                end_time=end_time.strftime('%Y-%m-%d'),
                limit=1000
            )
            
            if df is not None and not df.empty:
                self._data_cache = df
                self._cache_timestamp = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting analysis data: {e}")
            self._data_cache = None # Invalidate cache on error
            return None
    
    def _check_entry_signals(self, analysis: Dict):
        """Check for entry signals."""
        try:
            # Check if we should enter a new position
            should_enter, entry_details = self.strategy.should_enter_position(
                analysis, len(self.positions)
            )
            
            if should_enter:
                # Check risk management
                if not self.risk_manager.should_allow_new_position(
                    len(self.positions), self.account_balance
                ):
                    self.logger.info("Risk management blocked new position")
                    return
                
                # Execute entry
                self._execute_entry(entry_details)
            
        except Exception as e:
            self.logger.error(f"Error checking entry signals: {e}")
    
    def _check_exit_signals(self, analysis: Dict):
        """Check for exit signals."""
        try:
            for position in self.positions[:]:  # Copy list to avoid modification during iteration
                should_exit, exit_details = self.strategy.should_exit_position(analysis, position)
                
                if should_exit:
                    self._execute_exit(position, exit_details)
            
        except Exception as e:
            self.logger.error(f"Error checking exit signals: {e}")
    
    def _execute_entry(self, entry_details: Dict):
        """Execute a trade entry."""
        try:
            current_price = entry_details['current_price']
            atr = entry_details['atr']
            
            # Determine direction (simplified - always long for now)
            direction = 'long'
            
            # Calculate position size
            stop_loss = self.risk_manager.calculate_stop_loss(current_price, atr, direction)
            position_size = self.risk_manager.calculate_position_size(
                self.account_balance, current_price, stop_loss
            )
            
            # Calculate take profit
            take_profit = self.risk_manager.calculate_take_profit(current_price, atr, direction)
            
            # Validate trade parameters
            is_valid, error_msg = self.risk_manager.validate_trade_parameters(
                current_price, stop_loss, take_profit, position_size
            )
            
            if not is_valid:
                self.logger.warning(f"Trade validation failed: {error_msg}")
                return
            
            # Create position
            position = {
                'id': f"pos_{int(time.time())}",
                'symbol': self.symbol,
                'direction': direction,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'entry_reasons': entry_details['reasons'],
                'confidence': entry_details['confidence']
            }
            
            # Execute trade
            if self.mode == 'live':
                success = self._execute_live_trade(position, 'entry')
            else:
                success = self._execute_paper_trade(position, 'entry')
            
            if success:
                self.positions.append(position)
                self.logger.info(f"Position opened: {position['id']}")
                
                # Send notification
                self.telegram_notifier.notify_trade_entry(position)
                
                # Log trade
                self._log_trade(position, 'entry')
            
        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            self.telegram_notifier.notify_error(str(e), "Trade entry")
    
    def _execute_exit(self, position: Dict, exit_details: Dict):
        """Execute a trade exit."""
        try:
            current_price = self.data_manager.get_current_price(self.symbol)
            
            # Calculate P&L
            if position['direction'] == 'long':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
            
            pnl_percent = pnl / (position['entry_price'] * position['size'])
            
            # Update position
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['pnl_percent'] = pnl_percent
            position['exit_reason'] = exit_details['reason']
            
            # Execute trade
            if self.mode == 'live':
                success = self._execute_live_trade(position, 'exit')
            else:
                success = self._execute_paper_trade(position, 'exit')
            
            if success:
                # Remove from positions
                self.positions.remove(position)
                
                # Update account balance
                self.account_balance += pnl
                self.total_pnl += pnl
                
                # Update statistics
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Record trade
                self.risk_manager.record_trade(pnl, position)
                self.trade_history.append(position)
                
                self.logger.info(f"Position closed: {position['id']}, P&L: ${pnl:,.2f}")
                
                # Send notification
                self.telegram_notifier.notify_trade_exit(position)
                
                # Log trade
                self._log_trade(position, 'exit')
            
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            self.telegram_notifier.notify_error(str(e), "Trade exit")
    
    def _execute_live_trade(self, position: Dict, action: str) -> bool:
        """Execute live trade on Binance."""
        try:
            # This is a placeholder for live trading
            # In a real implementation, you would use the Binance API
            self.logger.info(f"Live trade {action}: {position['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing live trade: {e}")
            return False
    
    def _execute_paper_trade(self, position: Dict, action: str) -> bool:
        """Execute paper trade (simulation)."""
        try:
            self.logger.info(f"Paper trade {action}: {position['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            return False
    
    def _update_trailing_stops(self, current_price: float):
        """Update trailing stops for all positions."""
        try:
            for position in self.positions:
                new_stop_loss = self.risk_manager.update_trailing_stop(
                    current_price,
                    position['entry_price'],
                    position['stop_loss'],
                    position['direction']
                )
                
                if new_stop_loss != position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
                    self.logger.info(f"Updated trailing stop for {position['id']}: {new_stop_loss:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")
    
    def _close_all_positions(self):
        """Close all open positions."""
        try:
            for position in self.positions[:]:
                exit_details = {
                    'reason': 'Bot shutdown',
                    'type': 'shutdown_exit'
                }
                self._execute_exit(position, exit_details)
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def _log_trade(self, position: Dict, action: str):
        """Log trade to file."""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'position_id': position['id'],
                'symbol': position['symbol'],
                'direction': position['direction'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'exit_price': position.get('exit_price'),
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'pnl': position.get('pnl'),
                'pnl_percent': position.get('pnl_percent'),
                'exit_reason': position.get('exit_reason'),
                'confidence': position.get('confidence'),
                'reasons': position.get('entry_reasons', [])
            }
            
            # Write to log file
            with open('trades.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of the trading bot."""
        try:
            return {
                'status': 'Running' if self.is_running else 'Stopped',
                'is_running': self.is_running,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'mode': self.mode,
                'account_balance': self.account_balance,
                'total_pnl': self.total_pnl,
                'open_positions': len(self.positions),
                'total_trades': len(self.trade_history),
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.winning_trades / max(1, len(self.trade_history)),
                'last_analysis': self.last_analysis,
                'positions': self.positions,
                'trade_history': self.trade_history[-10:]  # Last 10 trades
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        try:
            if not self.trade_history:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0
                }
            
            # Calculate metrics
            returns = [trade['pnl_percent'] for trade in self.trade_history if trade.get('pnl_percent') is not None]
            total_return = sum(returns)
            avg_return = np.mean(returns) if returns else 0
            std_return = np.std(returns) if returns else 0
            
            # Sharpe ratio (simplified)
            sharpe_ratio = avg_return / max(std_return, 0.001) if std_return > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade': avg_return,
                'best_trade': max(returns) if returns else 0,
                'worst_trade': min(returns) if returns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def get_balance(self) -> Dict:
        """
        Gets the current account balance.
        For paper trading, it returns the simulated balance.
        For live trading, it would fetch from the exchange.
        """
        if self.mode == 'paper':
            # For paper trading, the 'free' and 'total' balance are the same.
            return {
                'free': self.account_balance,
                'total': self.account_balance
            }
        else:
            # For live trading, fetch the real balance.
            try:
                account_info = self.data_manager.get_account_info()
                usdt_balance = {'free': 0, 'total': 0, 'asset': 'USDT'}
                for balance in account_info.get('balances', []):
                    if balance['asset'] == 'USDT':
                        free = float(balance['free'])
                        locked = float(balance['locked'])
                        usdt_balance['free'] = free
                        usdt_balance['total'] = free + locked
                        break
                return usdt_balance
            except Exception as e:
                self.logger.error(f"Could not fetch live account balance: {e}")
                return {'free': 0, 'total': 0}

    def set_symbol(self, symbol: str):
        """Set a new trading symbol."""
        if symbol != self.symbol:
            self.logger.info(f"Changing symbol from {self.symbol} to {symbol}")
            was_running = self.is_running
            if was_running:
                self.stop()
            
            self.symbol = symbol
            if self.config and hasattr(self.config, 'config'):
                 self.config.config['trading']['symbol'] = symbol
            
            if was_running:
                # Give some time for old websocket to close
                time.sleep(1)
                self.start() 
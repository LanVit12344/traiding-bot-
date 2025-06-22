"""
Backtesting module for historical strategy testing.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, config, data_manager, indicator_engine, ai_model, 
                 news_analyzer, strategy, risk_manager):
        """Initialize backtester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.indicator_engine = indicator_engine
        self.ai_model = ai_model
        self.news_analyzer = news_analyzer
        self.strategy = strategy
        self.risk_manager = risk_manager
        
        # Backtesting parameters
        self.initial_balance = config.get('backtest.initial_balance', 10000)
        self.commission = config.get('backtest.commission', 0.001)
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.positions = []
    
    def run(self, start_date: str, end_date: str) -> Dict:
        """
        Run backtest for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Reset results
            self.trades = []
            self.equity_curve = []
            self.positions = []
            
            # Get historical data
            df = self.data_manager.get_historical_data(
                symbol=self.config.get('trading.symbol'),
                interval=self.config.get('trading.timeframe'),
                start_time=start_date,
                end_date=end_date,
                limit=10000
            )
            
            if df is None or df.empty:
                raise ValueError("No historical data available for backtest")
            
            # Calculate indicators
            df = self.indicator_engine.calculate_all_indicators(df)
            
            # Run simulation
            results = self._run_simulation(df)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(results)
            
            self.logger.info("Backtest completed successfully")

            # Return a clean dictionary with DataFrames
            return {
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': self.initial_balance,
                'final_balance': results['final_balance'],
                'total_return': performance.get('total_return', 0),
                'total_trades': len(self.trades),
                'winning_trades': performance.get('winning_trades', 0),
                'losing_trades': performance.get('losing_trades', 0),
                'win_rate': performance.get('win_rate', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'avg_trade': performance.get('avg_trade', 0),
                'best_trade': performance.get('best_trade', 0),
                'worst_trade': performance.get('worst_trade', 0),
                'trades': pd.DataFrame(self.trades),
                'equity_curve': pd.DataFrame(self.equity_curve).set_index('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    def _run_simulation(self, df: pd.DataFrame) -> Dict:
        """Run the trading simulation."""
        try:
            balance = self.initial_balance
            current_positions = []
            
            # Track equity curve
            self.equity_curve = []
            
            for i in range(50, len(df)):  # Start after enough data for indicators
                current_data = df.iloc[:i+1]
                current_price = current_data.iloc[-1]['close']
                current_time = current_data.index[-1]
                
                # Update existing positions
                current_positions = self._update_positions(current_positions, current_data)
                
                # Calculate current equity
                equity = balance + sum(pos['unrealized_pnl'] for pos in current_positions)
                self.equity_curve.append({
                    'timestamp': current_time,
                    'equity': equity,
                    'balance': balance,
                    'positions': len(current_positions)
                })
                
                # Check for exit signals
                if current_positions:
                    self._check_exit_signals(current_positions, current_data, balance)
                
                # Check for entry signals
                if len(current_positions) < self.config.get('trading.max_positions', 3):
                    self._check_entry_signals(current_data, current_positions, balance)
            
            # Close any remaining positions
            for position in current_positions:
                final_price = df.iloc[-1]['close']
                pnl = self._calculate_pnl(position, final_price)
                balance += pnl
                
                # Record trade
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'pnl_percent': pnl / (position['entry_price'] * position['size']),
                    'exit_reason': 'backtest_end'
                }
                self.trades.append(trade)
            
            return {
                'final_balance': balance,
                'total_trades': len(self.trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            raise
    
    def _check_entry_signals(self, current_data: pd.DataFrame, 
                           current_positions: List, balance: float):
        """Check for entry signals."""
        try:
            # Perform market analysis
            analysis = self.strategy.analyze_market(current_data)
            
            # Check if we should enter a position
            should_enter, entry_details = self.strategy.should_enter_position(
                analysis, len(current_positions)
            )
            
            if should_enter:
                # Calculate position size
                current_price = entry_details['current_price']
                atr = entry_details['atr']
                
                # Calculate stop loss and take profit
                stop_loss = self.risk_manager.calculate_stop_loss(current_price, atr, 'long')
                take_profit = self.risk_manager.calculate_take_profit(current_price, atr, 'long')
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    balance, current_price, stop_loss
                )
                
                # Check if we have enough balance
                required_balance = position_size * current_price * (1 + self.commission)
                if required_balance <= balance:
                    # Create position
                    position = {
                        'id': f"pos_{len(self.trades)}",
                        'entry_time': current_data.index[-1],
                        'entry_price': current_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'unrealized_pnl': 0.0
                    }
                    
                    current_positions.append(position)
                    
                    # Deduct balance
                    balance -= required_balance
                    
                    self.logger.info(f"Position opened: {position['id']} at ${current_price:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error checking entry signals: {e}")
    
    def _check_exit_signals(self, current_positions: List, current_data: pd.DataFrame, balance: float):
        """Check for exit signals."""
        try:
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.index[-1]
            
            # Perform market analysis
            analysis = self.strategy.analyze_market(current_data)
            
            for position in current_positions[:]:  # Copy list to avoid modification during iteration
                should_exit, exit_details = self.strategy.should_exit_position(analysis, position)
                
                if should_exit:
                    # Calculate P&L
                    pnl = self._calculate_pnl(position, current_price)
                    balance += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': pnl / (position['entry_price'] * position['size']),
                        'exit_reason': exit_details['reason']
                    }
                    self.trades.append(trade)
                    
                    # Remove position
                    current_positions.remove(position)
                    
                    self.logger.info(f"Position closed: {position['id']}, P&L: ${pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error checking exit signals: {e}")
    
    def _update_positions(self, positions: List, current_data: pd.DataFrame) -> List:
        """Update position unrealized P&L and check stop loss/take profit."""
        try:
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.index[-1]
            
            for position in positions[:]:  # Copy list to avoid modification during iteration
                # Update unrealized P&L
                position['unrealized_pnl'] = self._calculate_pnl(position, current_price)
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    # Stop loss hit
                    pnl = self._calculate_pnl(position, position['stop_loss'])
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': pnl / (position['entry_price'] * position['size']),
                        'exit_reason': 'stop_loss'
                    }
                    self.trades.append(trade)
                    
                    # Remove position
                    positions.remove(position)
                    
                    self.logger.info(f"Stop loss hit: {position['id']}, P&L: ${pnl:.2f}")
                
                # Check take profit
                elif current_price >= position['take_profit']:
                    # Take profit hit
                    pnl = self._calculate_pnl(position, position['take_profit'])
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': pnl / (position['entry_price'] * position['size']),
                        'exit_reason': 'take_profit'
                    }
                    self.trades.append(trade)
                    
                    # Remove position
                    positions.remove(position)
                    
                    self.logger.info(f"Take profit hit: {position['id']}, P&L: ${pnl:.2f}")
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return positions
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L for a position."""
        try:
            # Calculate gross P&L
            gross_pnl = (exit_price - position['entry_price']) * position['size']
            
            # Calculate commission
            entry_commission = position['entry_price'] * position['size'] * self.commission
            exit_commission = exit_price * position['size'] * self.commission
            total_commission = entry_commission + exit_commission
            
            # Net P&L
            net_pnl = gross_pnl - total_commission
            
            return net_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results."""
        try:
            if not self.trades:
                return {
                    'total_return': 0.0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0
                }
            
            # Basic metrics
            total_return = (results['final_balance'] - self.initial_balance) / self.initial_balance
            winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            losing_trades = sum(1 for trade in self.trades if trade['pnl'] < 0)
            win_rate = winning_trades / len(self.trades) if self.trades else 0
            
            # Trade metrics
            pnls = [trade['pnl'] for trade in self.trades]
            avg_trade = np.mean(pnls)
            best_trade = max(pnls)
            worst_trade = min(pnls)
            
            # Sharpe ratio (simplified)
            returns = [trade['pnl_percent'] for trade in self.trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / max(std_return, 0.001) if std_return > 0 else 0
            
            # Maximum drawdown
            if self.equity_curve:
                equities = [point['equity'] for point in self.equity_curve]
                cumulative_returns = [(equity - self.initial_balance) / self.initial_balance for equity in equities]
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0.0
            
            return {
                'total_return': total_return,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade': avg_trade,
                'best_trade': best_trade,
                'worst_trade': worst_trade
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades."""
        try:
            if not self.trades:
                return pd.DataFrame()
            
            df_trades = pd.DataFrame(self.trades)
            df_trades['duration'] = pd.to_datetime(df_trades['exit_time']) - pd.to_datetime(df_trades['entry_time'])
            
            return df_trades
            
        except Exception as e:
            self.logger.error(f"Error getting trade summary: {e}")
            return pd.DataFrame()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve data."""
        try:
            if not self.equity_curve:
                return pd.DataFrame()
            
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
            df_equity.set_index('timestamp', inplace=True)
            
            return df_equity
            
        except Exception as e:
            self.logger.error(f"Error getting equity curve: {e}")
            return pd.DataFrame() 
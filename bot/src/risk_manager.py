"""
Risk management module for position sizing and risk control.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

class RiskManager:
    """Manages risk and position sizing for trading."""
    
    def __init__(self, config):
        """Initialize risk manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = config.get('trading.max_position_size', 0.02)
        self.max_positions = config.get('trading.max_positions', 3)
        self.max_daily_loss = config.get('risk.max_daily_loss', 0.05)
        self.stop_loss_atr_multiplier = config.get('risk.stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_multiplier = config.get('risk.take_profit_atr_multiplier', 3.0)
        self.trailing_stop = config.get('risk.trailing_stop', True)
        self.trailing_stop_distance = config.get('risk.trailing_stop_distance', 0.01)
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
    
    def calculate_position_size(self, account_balance: float, 
                              current_price: float,
                              stop_loss_price: float,
                              sentiment_adjustment: float = 1.0) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            stop_loss_price: Stop loss price
            sentiment_adjustment: Adjustment factor from sentiment analysis
            
        Returns:
            Position size in base asset units
        """
        try:
            # Calculate risk per trade (1% of account)
            risk_per_trade = account_balance * 0.01
            
            # Calculate price risk
            price_risk = abs(current_price - stop_loss_price)
            if price_risk == 0:
                self.logger.warning("Price risk is zero, using minimum position size")
                return account_balance * 0.001 / current_price
            
            # Calculate position size based on risk
            position_size = risk_per_trade / price_risk
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / current_price
            
            # Apply sentiment adjustment
            adjusted_position_size = position_size * sentiment_adjustment
            
            # Take the minimum of calculated and maximum position size
            final_position_size = min(adjusted_position_size, max_position_size)
            
            # Ensure minimum position size
            min_position_value = account_balance * 0.001  # 0.1% minimum
            min_position_size = min_position_value / current_price
            
            if final_position_size < min_position_size:
                final_position_size = min_position_size
            
            self.logger.info(f"Position size calculated: {final_position_size:.6f} "
                           f"(Risk: ${risk_per_trade:.2f}, Max: {max_position_size:.6f})")
            
            return final_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, 
                          atr: float, 
                          direction: str) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        try:
            atr_distance = atr * self.stop_loss_atr_multiplier
            
            if direction == 'long':
                stop_loss = entry_price - atr_distance
            elif direction == 'short':
                stop_loss = entry_price + atr_distance
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
            self.logger.info(f"Stop loss calculated: {stop_loss:.6f} "
                           f"(ATR: {atr:.6f}, Distance: {atr_distance:.6f})")
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.95 if direction == 'long' else entry_price * 1.05
    
    def calculate_take_profit(self, entry_price: float,
                            atr: float,
                            direction: str) -> float:
        """
        Calculate take profit price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        try:
            atr_distance = atr * self.take_profit_atr_multiplier
            
            if direction == 'long':
                take_profit = entry_price + atr_distance
            elif direction == 'short':
                take_profit = entry_price - atr_distance
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
            self.logger.info(f"Take profit calculated: {take_profit:.6f} "
                           f"(ATR: {atr:.6f}, Distance: {atr_distance:.6f})")
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return entry_price * 1.05 if direction == 'long' else entry_price * 0.95
    
    def update_trailing_stop(self, current_price: float,
                           entry_price: float,
                           current_stop_loss: float,
                           direction: str) -> float:
        """
        Update trailing stop loss.
        
        Args:
            current_price: Current market price
            entry_price: Original entry price
            current_stop_loss: Current stop loss price
            direction: 'long' or 'short'
            
        Returns:
            Updated stop loss price
        """
        try:
            if not self.trailing_stop:
                return current_stop_loss
            
            if direction == 'long':
                # For long positions, trail below current price
                new_stop_loss = current_price * (1 - self.trailing_stop_distance)
                
                # Only update if new stop loss is higher than current
                if new_stop_loss > current_stop_loss:
                    self.logger.info(f"Trailing stop updated: {current_stop_loss:.6f} -> {new_stop_loss:.6f}")
                    return new_stop_loss
                    
            elif direction == 'short':
                # For short positions, trail above current price
                new_stop_loss = current_price * (1 + self.trailing_stop_distance)
                
                # Only update if new stop loss is lower than current
                if new_stop_loss < current_stop_loss:
                    self.logger.info(f"Trailing stop updated: {current_stop_loss:.6f} -> {new_stop_loss:.6f}")
                    return new_stop_loss
            
            return current_stop_loss
            
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")
            return current_stop_loss
    
    def check_daily_loss_limit(self, trade_pnl: float) -> bool:
        """
        Check if daily loss limit would be exceeded.
        
        Args:
            trade_pnl: P&L of the current trade
            
        Returns:
            True if trade should be allowed, False if limit would be exceeded
        """
        try:
            # Reset daily tracking if it's a new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_pnl = 0.0
                self.daily_trades = []
                self.last_reset_date = current_date
                self.logger.info("Daily P&L tracking reset")
            
            # Check if adding this trade would exceed daily loss limit
            projected_daily_pnl = self.daily_pnl + trade_pnl
            
            # Get account balance for percentage calculation
            # This would need to be passed from the main trading logic
            account_balance = 10000  # Placeholder - should be passed as parameter
            
            daily_loss_percentage = abs(min(0, projected_daily_pnl)) / account_balance
            
            if daily_loss_percentage > self.max_daily_loss:
                self.logger.warning(f"Daily loss limit would be exceeded: "
                                  f"{daily_loss_percentage:.2%} > {self.max_daily_loss:.2%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
            return True  # Allow trade if check fails
    
    def record_trade(self, trade_pnl: float, trade_details: Dict):
        """
        Record a completed trade for daily tracking.
        
        Args:
            trade_pnl: P&L of the trade
            trade_details: Additional trade information
        """
        try:
            self.daily_pnl += trade_pnl
            self.daily_trades.append({
                'timestamp': datetime.now(),
                'pnl': trade_pnl,
                'details': trade_details
            })
            
            self.logger.info(f"Trade recorded: P&L ${trade_pnl:.2f}, "
                           f"Daily total: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def get_risk_metrics(self, account_balance: float) -> Dict:
        """
        Get current risk metrics.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            daily_loss_percentage = abs(min(0, self.daily_pnl)) / account_balance
            remaining_daily_loss = (self.max_daily_loss - daily_loss_percentage) * account_balance
            
            return {
                'daily_pnl': self.daily_pnl,
                'daily_loss_percentage': daily_loss_percentage,
                'remaining_daily_loss': remaining_daily_loss,
                'max_daily_loss': self.max_daily_loss,
                'max_position_size': self.max_position_size,
                'max_positions': self.max_positions,
                'total_trades_today': len(self.daily_trades),
                'last_reset_date': self.last_reset_date.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def should_allow_new_position(self, current_positions: int,
                                account_balance: float,
                                estimated_trade_pnl: float = 0.0) -> bool:
        """
        Check if a new position should be allowed.
        
        Args:
            current_positions: Number of current open positions
            account_balance: Current account balance
            estimated_trade_pnl: Estimated P&L of the new trade (negative for risk)
            
        Returns:
            True if new position should be allowed
        """
        try:
            # Check position count limit
            if current_positions >= self.max_positions:
                self.logger.warning(f"Maximum positions reached: {current_positions}")
                return False
            
            # Check daily loss limit
            if not self.check_daily_loss_limit(estimated_trade_pnl):
                return False
            
            # Check if account has sufficient balance
            min_balance_required = account_balance * 0.1  # 10% minimum balance
            if account_balance < min_balance_required:
                self.logger.warning(f"Insufficient balance: ${account_balance:.2f} < ${min_balance_required:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position allowance: {e}")
            return False
    
    def calculate_risk_reward_ratio(self, entry_price: float,
                                  stop_loss_price: float,
                                  take_profit_price: float) -> float:
        """
        Calculate risk-reward ratio for a trade.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Risk-reward ratio
        """
        try:
            risk = abs(entry_price - stop_loss_price)
            reward = abs(take_profit_price - entry_price)
            
            if risk == 0:
                return 0.0
            
            ratio = reward / risk
            self.logger.info(f"Risk-reward ratio: {ratio:.2f}")
            
            return ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0.0
    
    def validate_trade_parameters(self, entry_price: float,
                                stop_loss_price: float,
                                take_profit_price: float,
                                position_size: float) -> Tuple[bool, str]:
        """
        Validate trade parameters before execution.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            position_size: Position size
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for valid prices
            if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
                return False, "Invalid price values"
            
            if position_size <= 0:
                return False, "Invalid position size"
            
            # Check stop loss logic
            if stop_loss_price >= entry_price and take_profit_price > entry_price:
                return False, "Invalid stop loss for long position"
            
            if stop_loss_price <= entry_price and take_profit_price < entry_price:
                return False, "Invalid stop loss for short position"
            
            # Check risk-reward ratio
            ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price)
            if ratio < 1.5:
                return False, f"Risk-reward ratio too low: {ratio:.2f}"
            
            return True, "Trade parameters valid"
            
        except Exception as e:
            self.logger.error(f"Error validating trade parameters: {e}")
            return False, f"Validation error: {str(e)}" 
"""
Telegram notification module for trading alerts.
"""
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

class TelegramNotifier:
    """Sends trading notifications via Telegram."""
    
    def __init__(self, config):
        """Initialize Telegram notifier."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Telegram configuration
        self.bot_token = config.get('telegram.bot_token')
        self.chat_id = config.get('telegram.chat_id')
        self.enabled = config.get('telegram.enabled', False)
        self.notify_events = config.get('telegram.notify_on', [])
        
        # Base URL for Telegram API
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram credentials not configured")
            self.enabled = False
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message via Telegram.
        
        Args:
            message: Message to send
            parse_mode: Message parsing mode ('HTML' or 'Markdown')
            
        Returns:
            True if message sent successfully
        """
        try:
            if not self.enabled:
                return False
            
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('ok'):
                self.logger.info("Telegram message sent successfully")
                return True
            else:
                self.logger.error(f"Telegram API error: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def notify_trade_entry(self, trade_details: Dict) -> bool:
        """Notify about trade entry."""
        try:
            if 'entry' not in self.notify_events:
                return True
            
            symbol = trade_details.get('symbol', 'Unknown')
            direction = trade_details.get('direction', 'Unknown')
            price = trade_details.get('entry_price', 0)
            size = trade_details.get('size', 0)
            confidence = trade_details.get('confidence', 0)
            reasons = trade_details.get('reasons', [])
            
            message = f"""
🚀 <b>TRADE ENTRY</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Direction:</b> {direction.upper()}
💰 <b>Price:</b> ${price:,.6f}
📏 <b>Size:</b> {size:.6f}
🎯 <b>Confidence:</b> {confidence:.1%}

📝 <b>Reasons:</b>
{chr(10).join(f"• {reason}" for reason in reasons[:5])}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying trade entry: {e}")
            return False
    
    def notify_trade_exit(self, trade_details: Dict) -> bool:
        """Notify about trade exit."""
        try:
            if 'exit' not in self.notify_events:
                return True
            
            symbol = trade_details.get('symbol', 'Unknown')
            direction = trade_details.get('direction', 'Unknown')
            entry_price = trade_details.get('entry_price', 0)
            exit_price = trade_details.get('exit_price', 0)
            size = trade_details.get('size', 0)
            pnl = trade_details.get('pnl', 0)
            pnl_percent = trade_details.get('pnl_percent', 0)
            exit_reason = trade_details.get('exit_reason', 'Unknown')
            
            # Determine emoji based on P&L
            if pnl > 0:
                emoji = "✅"
                result = "PROFIT"
            elif pnl < 0:
                emoji = "❌"
                result = "LOSS"
            else:
                emoji = "➖"
                result = "BREAKEVEN"
            
            message = f"""
{emoji} <b>TRADE EXIT - {result}</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Direction:</b> {direction.upper()}
💰 <b>Entry Price:</b> ${entry_price:,.6f}
💵 <b>Exit Price:</b> ${exit_price:,.6f}
📏 <b>Size:</b> {size:.6f}

💸 <b>P&L:</b> ${pnl:,.2f} ({pnl_percent:+.2%})
📋 <b>Exit Reason:</b> {exit_reason}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying trade exit: {e}")
            return False
    
    def notify_stop_loss(self, trade_details: Dict) -> bool:
        """Notify about stop loss hit."""
        try:
            if 'stop_loss' not in self.notify_events:
                return True
            
            symbol = trade_details.get('symbol', 'Unknown')
            direction = trade_details.get('direction', 'Unknown')
            entry_price = trade_details.get('entry_price', 0)
            stop_price = trade_details.get('stop_price', 0)
            size = trade_details.get('size', 0)
            pnl = trade_details.get('pnl', 0)
            
            message = f"""
🛑 <b>STOP LOSS HIT</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Direction:</b> {direction.upper()}
💰 <b>Entry Price:</b> ${entry_price:,.6f}
🛑 <b>Stop Price:</b> ${stop_price:,.6f}
📏 <b>Size:</b> {size:.6f}

💸 <b>P&L:</b> ${pnl:,.2f}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying stop loss: {e}")
            return False
    
    def notify_take_profit(self, trade_details: Dict) -> bool:
        """Notify about take profit hit."""
        try:
            if 'take_profit' not in self.notify_events:
                return True
            
            symbol = trade_details.get('symbol', 'Unknown')
            direction = trade_details.get('direction', 'Unknown')
            entry_price = trade_details.get('entry_price', 0)
            take_profit_price = trade_details.get('take_profit_price', 0)
            size = trade_details.get('size', 0)
            pnl = trade_details.get('pnl', 0)
            
            message = f"""
🎯 <b>TAKE PROFIT HIT</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Direction:</b> {direction.upper()}
💰 <b>Entry Price:</b> ${entry_price:,.6f}
🎯 <b>Take Profit Price:</b> ${take_profit_price:,.6f}
📏 <b>Size:</b> {size:.6f}

💸 <b>P&L:</b> ${pnl:,.2f}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying take profit: {e}")
            return False
    
    def notify_error(self, error_message: str, context: str = "") -> bool:
        """Notify about errors."""
        try:
            if 'error' not in self.notify_events:
                return True
            
            message = f"""
⚠️ <b>TRADING BOT ERROR</b>

🔍 <b>Context:</b> {context}
❌ <b>Error:</b> {error_message}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying error: {e}")
            return False
    
    def notify_daily_summary(self, summary: Dict) -> bool:
        """Send daily trading summary."""
        try:
            total_trades = summary.get('total_trades', 0)
            winning_trades = summary.get('winning_trades', 0)
            losing_trades = summary.get('losing_trades', 0)
            total_pnl = summary.get('total_pnl', 0)
            win_rate = summary.get('win_rate', 0)
            
            # Determine emoji based on performance
            if total_pnl > 0:
                emoji = "📈"
                performance = "PROFITABLE"
            elif total_pnl < 0:
                emoji = "📉"
                performance = "LOSS"
            else:
                emoji = "➖"
                performance = "BREAKEVEN"
            
            message = f"""
{emoji} <b>DAILY TRADING SUMMARY</b>

📊 <b>Performance:</b> {performance}
💰 <b>Total P&L:</b> ${total_pnl:,.2f}
📈 <b>Total Trades:</b> {total_trades}
✅ <b>Winning Trades:</b> {winning_trades}
❌ <b>Losing Trades:</b> {losing_trades}
🎯 <b>Win Rate:</b> {win_rate:.1%}

📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying daily summary: {e}")
            return False
    
    def notify_market_alert(self, alert_type: str, details: Dict) -> bool:
        """Send market alerts."""
        try:
            if alert_type == 'high_volatility':
                message = f"""
⚠️ <b>HIGH VOLATILITY ALERT</b>

📊 <b>Symbol:</b> {details.get('symbol', 'Unknown')}
📈 <b>Volatility:</b> {details.get('volatility', 0):.2%}
💰 <b>Current Price:</b> ${details.get('price', 0):,.6f}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """.strip()
            
            elif alert_type == 'trend_change':
                message = f"""
🔄 <b>TREND CHANGE ALERT</b>

📊 <b>Symbol:</b> {details.get('symbol', 'Unknown')}
📈 <b>Previous Trend:</b> {details.get('previous_trend', 'Unknown')}
📉 <b>New Trend:</b> {details.get('new_trend', 'Unknown')}
💰 <b>Price:</b> ${details.get('price', 0):,.6f}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """.strip()
            
            else:
                message = f"""
🔔 <b>MARKET ALERT</b>

📊 <b>Type:</b> {alert_type}
📝 <b>Details:</b> {json.dumps(details, indent=2)}

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying market alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection."""
        try:
            if not self.enabled:
                self.logger.warning("Telegram notifications disabled")
                return False
            
            message = f"""
🤖 <b>TRADING BOT CONNECTION TEST</b>

✅ Telegram notifications are working!
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            success = self.send_message(message)
            if success:
                self.logger.info("Telegram connection test successful")
            else:
                self.logger.error("Telegram connection test failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error testing Telegram connection: {e}")
            return False 
"""
Configuration management for the trading bot.
"""
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the trading bot."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file and environment variables."""
        load_dotenv()
        self.config_path = config_path
        self.config = self._load_config()
        self._load_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _load_env_vars(self):
        """Load sensitive data from environment variables."""
        self.config['binance'] = {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret_key': os.getenv('BINANCE_SECRET_KEY')
        }
        
        self.config['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
        self.config['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
        
        self.config['openai'] = {
            'api_key': os.getenv('OPENAI_API_KEY')
        }
        
        self.config['news']['cryptopanic_api_key'] = os.getenv('CRYPTOPANIC_API_KEY')
        self.config['news']['coindesk_api_key'] = os.getenv('COINDESK_API_KEY')
        
        self.config['environment'] = os.getenv('ENVIRONMENT', 'development')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get('trading', {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.config.get('risk', {})
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """Get technical indicators configuration."""
        return self.config.get('indicators', {})
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI model configuration."""
        return self.config.get('ai', {})
    
    def get_news_config(self) -> Dict[str, Any]:
        """Get news and sentiment configuration."""
        return self.config.get('news', {})
    
    def get_telegram_config(self) -> Dict[str, Any]:
        """Get Telegram configuration."""
        return self.config.get('telegram', {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get('dashboard', {})
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config.get('backtest', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def is_live_mode(self) -> bool:
        """Check if bot is running in live mode."""
        return self.get('trading.mode') == 'live'
    
    def is_paper_mode(self) -> bool:
        """Check if bot is running in paper mode."""
        return self.get('trading.mode') == 'paper'
    
    def is_backtest_mode(self) -> bool:
        """Check if bot is running in backtest mode."""
        return self.get('trading.mode') == 'backtest'
    
    def set(self, key: str, value):
        """Set a value in the config using dot notation."""
        try:
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            self.save()
        except Exception as e:
            print(f"Error setting config value: {e}")
    
    def save(self):
        """Save the current configuration back to the file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}") 
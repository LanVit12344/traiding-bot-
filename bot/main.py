"""
Main entry point for the trading bot.
"""
import sys
import os
import logging
import argparse
from datetime import datetime
import signal
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.data_manager import DataManager
from src.indicator_engine import IndicatorEngine
from src.ai_model import AIModel
from src.news_sentiment import NewsSentimentAnalyzer
from src.strategy import Strategy
from src.risk_manager import RiskManager
from src.telegram_notifier import TelegramNotifier
from src.trader import Trader

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the trading bot."""
        self.config_path = config_path
        self.config = None
        self.trader = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trading_bot.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all trading bot components."""
        try:
            self.logger.info("Initializing trading bot components...")
            
            # Load configuration
            self.config = Config(self.config_path)
            self.logger.info("Configuration loaded")
            
            # Initialize data manager
            self.data_manager = DataManager(self.config)
            self.logger.info("Data manager initialized")
            
            # Initialize indicator engine
            self.indicator_engine = IndicatorEngine(self.config)
            self.logger.info("Indicator engine initialized")
            
            # Initialize AI model
            self.ai_model = AIModel(self.config)
            self.logger.info("AI model initialized")
            
            # Initialize news sentiment analyzer
            self.news_analyzer = NewsSentimentAnalyzer(self.config)
            self.logger.info("News sentiment analyzer initialized")
            
            # Initialize strategy
            self.strategy = Strategy(
                self.config,
                self.indicator_engine,
                self.ai_model,
                self.news_analyzer
            )
            self.logger.info("Strategy initialized")
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)
            self.logger.info("Risk manager initialized")
            
            # Initialize Telegram notifier
            self.telegram_notifier = TelegramNotifier(self.config)
            self.logger.info("Telegram notifier initialized")
            
            # Initialize trader
            self.trader = Trader(
                self.config,
                self.data_manager,
                self.strategy,
                self.risk_manager,
                self.telegram_notifier
            )
            self.logger.info("Trader initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the trading bot."""
        try:
            self.logger.info("Starting trading bot...")
            
            # Start the trader
            if not self.trader.start():
                self.logger.error("Failed to start trader")
                return False
            
            self.logger.info("Trading bot started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            return False
    
    def stop(self):
        """Stop the trading bot."""
        try:
            self.logger.info("Stopping trading bot...")
            
            if self.trader:
                self.trader.stop()
            
            self.logger.info("Trading bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {e}")
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run backtesting."""
        try:
            self.logger.info(f"Running backtest from {start_date} to {end_date}")
            
            # Import backtesting module
            from src.backtester import Backtester
            
            # Initialize backtester
            backtester = Backtester(
                self.config,
                self.data_manager,
                self.indicator_engine,
                self.ai_model,
                self.news_analyzer,
                self.strategy,
                self.risk_manager
            )
            
            # Run backtest
            results = backtester.run(start_date, end_date)
            
            # Display results
            self._display_backtest_results(results)
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
    
    def _display_backtest_results(self, results: dict):
        """Display backtest results."""
        try:
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            
            print(f"Period: {results.get('start_date')} to {results.get('end_date')}")
            print(f"Initial Balance: ${results.get('initial_balance', 0):,.2f}")
            print(f"Final Balance: ${results.get('final_balance', 0):,.2f}")
            print(f"Total Return: {results.get('total_return', 0):.2%}")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Winning Trades: {results.get('winning_trades', 0)}")
            print(f"Losing Trades: {results.get('losing_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"Average Trade: ${results.get('avg_trade', 0):,.2f}")
            print(f"Best Trade: ${results.get('best_trade', 0):,.2f}")
            print(f"Worst Trade: ${results.get('worst_trade', 0):,.2f}")
            
            print("\n" + "="*50)
            
        except Exception as e:
            self.logger.error(f"Error displaying backtest results: {e}")
    
    def train_ai_model(self):
        """Train the AI model with historical data."""
        try:
            self.logger.info("Training AI model...")
            
            # Get historical data for training
            end_time = datetime.now()
            start_time = end_time.replace(year=end_time.year - 1)  # 1 year of data
            
            df = self.data_manager.get_historical_data(
                symbol=self.config.get('trading.symbol'),
                interval=self.config.get('trading.timeframe'),
                start_time=start_time.strftime('%Y-%m-%d'),
                end_time=end_time.strftime('%Y-%m-%d'),
                limit=5000
            )
            
            if df is None or df.empty:
                self.logger.error("No historical data available for training")
                return False
            
            # Calculate indicators
            df = self.indicator_engine.calculate_all_indicators(df)
            
            # Train AI model
            self.ai_model.train(df)
            
            self.logger.info("AI model training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training AI model: {e}")
            return False
    
    def test_connections(self):
        """Test all connections."""
        try:
            self.logger.info("Testing connections...")
            
            # Test Binance connection
            try:
                price = self.data_manager.get_current_price(self.config.get('trading.symbol'))
                self.logger.info(f"Binance connection: OK (Price: ${price:,.6f})")
            except Exception as e:
                self.logger.error(f"Binance connection failed: {e}")
            
            # Test Telegram connection
            if self.telegram_notifier.enabled:
                if self.telegram_notifier.test_connection():
                    self.logger.info("Telegram connection: OK")
                else:
                    self.logger.error("Telegram connection failed")
            else:
                self.logger.info("Telegram notifications disabled")
            
            # Test sentiment analysis
            try:
                sentiment = self.news_analyzer.get_overall_sentiment()
                self.logger.info(f"Sentiment analysis: OK (Score: {sentiment[0]:.2f})")
            except Exception as e:
                self.logger.error(f"Sentiment analysis failed: {e}")
            
            self.logger.info("Connection tests completed")
            
        except Exception as e:
            self.logger.error(f"Error testing connections: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest', 'dashboard', 'train', 'test'], 
                       default='paper', help='Bot mode')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'dashboard':
            print("\nTo run the dashboard, please use the following command:")
            print("streamlit run src/dashboard.py\n")
            return
            
        # Create trading bot
        bot = TradingBot(args.config)
        
        if args.mode == 'backtest':
            # Run backtest
            if not args.start_date or not args.end_date:
                print("Error: Start date and end date required for backtest mode")
                return
            
            bot.run_backtest(args.start_date, args.end_date)
        
        elif args.mode == 'train':
            # Train AI model
            bot.train_ai_model()
        
        elif args.mode == 'test':
            # Test connections
            bot.test_connections()
        
        else:
            # Run in live/paper mode
            if bot.start():
                print("Trading bot started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping trading bot...")
                    bot.stop()
            else:
                print("Failed to start trading bot")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
"""
News and sentiment analysis module for crypto trading.
"""
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import re

# Sentiment analysis imports
try:
    import openai
    from transformers import pipeline
    HAS_SENTIMENT_LIBS = True
except ImportError:
    HAS_SENTIMENT_LIBS = False

class NewsSentimentAnalyzer:
    """Analyzes crypto news and sentiment for trading decisions."""
    
    def __init__(self, config):
        """Initialize news and sentiment analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.openai_client = None
        self.sentiment_pipeline = None
        
        # Initialize sentiment analysis
        self._init_sentiment_analysis()
        
        # News sources configuration
        self.news_sources = config.get('news.sources', ['coindesk', 'cryptopanic'])
        self.sentiment_threshold = config.get('news.sentiment_threshold', -0.3)
        self.check_interval = config.get('news.check_interval', 30)
        
        # API keys
        self.openai_api_key = config.get('openai.api_key')
        self.cryptopanic_api_key = config.get('news.cryptopanic_api_key')
        self.coindesk_api_key = config.get('news.coindesk_api_key')
        
        # Cache for news data
        self.news_cache = {}
        self.last_check = None
        self._sentiment_cache = None
        self._sentiment_cache_time = None
    
    def _init_sentiment_analysis(self):
        """Initialize sentiment analysis tools."""
        try:
            if not HAS_SENTIMENT_LIBS:
                self.logger.warning("Sentiment analysis libraries not available")
                return
            
            # Initialize OpenAI client
            if self.config.get('openai.api_key'):
                openai.api_key = self.config.get('openai.api_key')
                self.openai_client = openai
                self.logger.info("OpenAI client initialized")
            
            # Initialize HuggingFace pipeline
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                self.logger.info("HuggingFace sentiment pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize HuggingFace pipeline: {e}")
                
        except Exception as e:
            self.logger.error(f"Error initializing sentiment analysis: {e}")
    
    def get_crypto_news(self, limit: int = 10) -> List[Dict]:
        """
        Fetch crypto news from multiple sources.
        
        Args:
            limit: Maximum number of news articles to fetch
            
        Returns:
            List of news articles with sentiment scores
        """
        try:
            all_news = []
            
            # Check cache first
            cache_key = f"news_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            # Fetch from different sources
            for source in self.news_sources:
                try:
                    if source == 'cryptopanic':
                        news = self._get_cryptopanic_news(limit)
                    elif source == 'coindesk':
                        news = self._get_coindesk_news(limit)
                    else:
                        continue
                    
                    all_news.extend(news)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching news from {source}: {e}")
            
            # Remove duplicates and limit results
            unique_news = self._remove_duplicates(all_news)
            limited_news = unique_news[:limit]
            
            # Analyze sentiment for each article
            for article in limited_news:
                article['sentiment'] = self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
            
            # Cache results
            self.news_cache[cache_key] = limited_news
            self.last_check = datetime.now()
            
            return limited_news
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto news: {e}")
            return []
    
    def _get_cryptopanic_news(self, limit: int) -> List[Dict]:
        """Fetch news from CryptoPanic API."""
        try:
            if not self.cryptopanic_api_key:
                self.logger.warning("CryptoPanic API key not configured")
                return []
            
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.cryptopanic_api_key,
                'filter': 'hot',
                'public': 'true',
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for post in data.get('results', []):
                article = {
                    'title': post.get('title', ''),
                    'description': post.get('metadata', {}).get('description', ''),
                    'url': post.get('url', ''),
                    'published_at': post.get('published_at', ''),
                    'source': 'cryptopanic',
                    'currencies': [curr.get('code') for curr in post.get('currencies', [])]
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching CryptoPanic news: {e}")
            return []
    
    def _get_coindesk_news(self, limit: int) -> List[Dict]:
        """Fetch news from CoinDesk API."""
        try:
            url = "https://api.coindesk.com/v1/news"
            params = {
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for post in data.get('posts', []):
                article = {
                    'title': post.get('title', ''),
                    'description': post.get('description', ''),
                    'url': post.get('url', ''),
                    'published_at': post.get('published_at', ''),
                    'source': 'coindesk'
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching CoinDesk news: {e}")
            return []
    
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity."""
        try:
            unique_articles = []
            seen_titles = set()
            
            for article in articles:
                title = article.get('title', '').lower()
                # Simple similarity check
                is_duplicate = any(
                    self._similarity(title, seen_title) > 0.8
                    for seen_title in seen_titles
                )
                
                if not is_duplicate:
                    unique_articles.append(article)
                    seen_titles.add(title)
            
            return unique_articles
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {e}")
            return articles
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        try:
            words1 = set(re.findall(r'\w+', text1))
            words2 = set(re.findall(r'\w+', text2))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            if not text.strip():
                return 0.0
            
            # Try OpenAI first
            if self.openai_client:
                try:
                    return self._analyze_sentiment_openai(text)
                except Exception as e:
                    self.logger.warning(f"OpenAI sentiment analysis failed: {e}")
            
            # Fallback to HuggingFace
            if self.sentiment_pipeline:
                try:
                    return self._analyze_sentiment_huggingface(text)
                except Exception as e:
                    self.logger.warning(f"HuggingFace sentiment analysis failed: {e}")
            
            # Final fallback to keyword-based
            return self._analyze_sentiment_keywords(text)
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _analyze_sentiment_openai(self, text: str) -> float:
        """Analyze sentiment using OpenAI API."""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analyzer. Analyze the sentiment of the given text and return a score between -1 (very negative) and 1 (very positive). Return only the number."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of this crypto-related text: {text[:500]}"
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Ensure score is between -1 and 1
            return max(-1, min(1, score))
            
        except Exception as e:
            self.logger.error(f"OpenAI sentiment analysis error: {e}")
            raise
    
    def _analyze_sentiment_huggingface(self, text: str) -> float:
        """Analyze sentiment using HuggingFace pipeline."""
        try:
            results = self.sentiment_pipeline(text[:512])  # Limit text length
            
            # Map labels to scores
            label_scores = {
                'LABEL_0': -1,  # Negative
                'LABEL_1': 0,   # Neutral
                'LABEL_2': 1    # Positive
            }
            
            # Get the highest scoring label
            best_result = max(results[0], key=lambda x: x['score'])
            score = label_scores.get(best_result['label'], 0)
            
            return score * best_result['score']  # Weight by confidence
            
        except Exception as e:
            self.logger.error(f"HuggingFace sentiment analysis error: {e}")
            raise
    
    def _analyze_sentiment_keywords(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        try:
            text_lower = text.lower()
            
            # Positive keywords
            positive_words = [
                'bullish', 'surge', 'rally', 'gain', 'profit', 'up', 'positive',
                'growth', 'adoption', 'partnership', 'launch', 'success', 'breakout'
            ]
            
            # Negative keywords
            negative_words = [
                'bearish', 'crash', 'drop', 'loss', 'down', 'negative', 'decline',
                'sell', 'dump', 'fear', 'panic', 'regulation', 'ban', 'hack'
            ]
            
            # Count occurrences
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate score
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            score = (positive_count - negative_count) / total_words
            return max(-1, min(1, score * 10))  # Scale and clamp
            
        except Exception as e:
            self.logger.error(f"Keyword sentiment analysis error: {e}")
            return 0.0
    
    def get_overall_sentiment(self) -> Tuple[float, str]:
        """
        Get overall sentiment of the market based on news.
        This method is cached to avoid excessive API calls.
        """
        try:
            # Check cache first
            cache_duration = timedelta(minutes=self.config.get('news.check_interval', 5))
            if self._sentiment_cache is not None and \
               self._sentiment_cache_time is not None and \
               datetime.now() - self._sentiment_cache_time < cache_duration:
                self.logger.debug("Returning cached sentiment.")
                return self._sentiment_cache

            news_articles = self.get_crypto_news()
            
            if not news_articles:
                return 0.0, "Neutral"
            
            # Calculate average sentiment
            total_sentiment = sum(article.get('sentiment', 0.0) for article in news_articles)
            avg_sentiment = total_sentiment / len(news_articles)
            
            # Describe sentiment
            if avg_sentiment > 0.3:
                description = "Strongly Positive"
            elif avg_sentiment > 0.1:
                description = "Positive"
            elif avg_sentiment < -0.3:
                description = "Strongly Negative"
            elif avg_sentiment < -0.1:
                description = "Negative"
            else:
                description = "Neutral"
            
            self.logger.info(f"Overall market sentiment: {description} (Score: {avg_sentiment:.2f})")

            # Cache the result
            self._sentiment_cache = (avg_sentiment, description)
            self._sentiment_cache_time = datetime.now()
            
            return self._sentiment_cache
            
        except Exception as e:
            self.logger.error(f"Error getting overall sentiment: {e}")
            return 0.0, "Neutral"
    
    def should_adjust_position_sizing(self) -> Tuple[bool, float]:
        """
        Determine if position sizing should be adjusted based on sentiment.
        
        Returns:
            Tuple of (should_adjust, adjustment_factor)
        """
        try:
            sentiment_score, _ = self.get_overall_sentiment()
            
            # Check if sentiment is extremely negative
            if sentiment_score < self.sentiment_threshold:
                # Reduce position sizing
                adjustment_factor = max(0.1, 1.0 + sentiment_score)  # Reduce by up to 90%
                return True, adjustment_factor
            
            # Check if sentiment is extremely positive
            elif sentiment_score > 0.5:
                # Slightly increase position sizing
                adjustment_factor = min(1.5, 1.0 + sentiment_score * 0.5)
                return True, adjustment_factor
            
            return False, 1.0
            
        except Exception as e:
            self.logger.error(f"Error checking position sizing adjustment: {e}")
            return False, 1.0
    
    def get_sentiment_summary(self) -> Dict:
        """
        Get a summary of current sentiment analysis.
        
        Returns:
            Dictionary with sentiment summary
        """
        try:
            news = self.get_crypto_news(limit=10)
            overall_sentiment, description = self.get_overall_sentiment()
            should_adjust, adjustment_factor = self.should_adjust_position_sizing()
            
            # Count sentiment categories
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            for article in news:
                sentiment = article.get('sentiment', 0)
                if sentiment > 0.1:
                    sentiment_counts['positive'] += 1
                elif sentiment < -0.1:
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_description': description,
                'should_adjust_position_sizing': should_adjust,
                'position_adjustment_factor': adjustment_factor,
                'sentiment_counts': sentiment_counts,
                'total_articles': len(news),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_description': 'neutral',
                'should_adjust_position_sizing': False,
                'position_adjustment_factor': 1.0,
                'sentiment_counts': {'positive': 0, 'neutral': 0, 'negative': 0},
                'total_articles': 0,
                'last_updated': datetime.now().isoformat()
            } 
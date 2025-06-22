"""
AI model for generating trading signals using machine learning.
"""
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

class AIModel:
    """AI model for generating trading signals."""
    
    def __init__(self, config):
        """Initialize AI model with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.model_type = config.get('ai.model_type', 'lstm')
        self.lookback_period = config.get('ai.lookback_period', 60)
        self.confidence_threshold = config.get('ai.confidence_threshold', 0.6)
        self.model_path = f"models/{self.model_type}_model.pkl"
        self.scaler_path = f"models/{self.model_type}_scaler.pkl"
        self._signal_cache = {}
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the AI model based on configuration."""
        try:
            if self.model_type == 'lstm':
                self._init_lstm_model()
            elif self.model_type == 'random_forest':
                self._init_random_forest_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            self.logger.error(f"Error initializing AI model: {e}")
            raise
    
    def _init_lstm_model(self):
        """Initialize LSTM model."""
        try:
            # Load existing model if available
            if os.path.exists(self.model_path.replace('.pkl', '.h5')):
                self.model = load_model(self.model_path.replace('.pkl', '.h5'))
                self.logger.info("Loaded existing LSTM model")
            else:
                # Create new LSTM model
                self.model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 16)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(3, activation='softmax')  # Buy, Sell, Hold
                ])
                
                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                self.logger.info("Created new LSTM model")
                
        except Exception as e:
            self.logger.error(f"Error initializing LSTM model: {e}")
            raise
    
    def _init_random_forest_model(self):
        """Initialize Random Forest model."""
        try:
            # Load existing model if available
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("Loaded existing Random Forest model")
            else:
                # Create new Random Forest model
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.logger.info("Created new Random Forest model")
                
        except Exception as e:
            self.logger.error(f"Error initializing Random Forest model: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Select features for training
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent',
                'ema_short', 'ema_long', 'atr'
            ]
            
            # Ensure all required columns exist
            available_columns = [col for col in feature_columns if col in df.columns]
            if len(available_columns) < 10:
                self.logger.warning(f"Limited features available: {available_columns}")
            
            # Prepare features
            features = df[available_columns].values
            
            # Create labels (future price direction)
            future_returns = df['close'].pct_change(5).shift(-5)  # 5-period future returns
            
            # Create categorical labels
            labels = np.zeros(len(features))
            labels[future_returns > 0.01] = 1  # Buy signal (1% gain)
            labels[future_returns < -0.01] = 2  # Sell signal (1% loss)
            # 0 = Hold
            
            # Remove NaN values
            valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
            features = features[valid_indices]
            labels = labels[valid_indices]
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        
        Args:
            features: Feature array
            labels: Label array
            
        Returns:
            Tuple of (sequences, labels)
        """
        try:
            sequences = []
            sequence_labels = []
            
            for i in range(self.lookback_period, len(features)):
                sequences.append(features[i-self.lookback_period:i])
                sequence_labels.append(labels[i])
            
            return np.array(sequences), np.array(sequence_labels)
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            raise
    
    def train(self, df: pd.DataFrame):
        """
        Train the AI model with historical data.
        
        Args:
            df: DataFrame with OHLCV data and indicators
        """
        try:
            self.logger.info("Starting AI model training...")
            
            # Prepare features and labels
            features, labels = self.prepare_features(df)
            
            if len(features) < 100:
                self.logger.warning("Insufficient data for training")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            if self.model_type == 'lstm':
                self._train_lstm(X_train, X_test, y_train, y_test)
            elif self.model_type == 'random_forest':
                self._train_random_forest(X_train, X_test, y_train, y_test)
            
            self.logger.info("AI model training completed")
            
        except Exception as e:
            self.logger.error(f"Error training AI model: {e}")
            raise
    
    def _train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM model."""
        try:
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
            X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
            
            # Scale features
            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train_seq.shape)
            
            X_test_scaled = self.scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1]))
            X_test_scaled = X_test_scaled.reshape(X_test_seq.shape)
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train_seq, 3)
            y_test_cat = to_categorical(y_test_seq, 3)
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train_cat,
                validation_data=(X_test_scaled, y_test_cat),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model
            self.model.save(self.model_path.replace('.pkl', '.h5'))
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test_seq, y_pred_classes)
            
            self.logger.info(f"LSTM model accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            raise
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model."""
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Random Forest model accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signal prediction.
        
        Args:
            df: DataFrame with recent OHLCV data and indicators
            
        Returns:
            Dictionary with prediction probabilities
        """
        try:
            if self.model is None:
                return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
            
            # Prepare features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent',
                'ema_short', 'ema_long', 'atr'
            ]
            
            available_columns = [col for col in feature_columns if col in df.columns]
            features = df[available_columns].tail(1).values
            
            if self.model_type == 'lstm':
                return self._predict_lstm(features)
            elif self.model_type == 'random_forest':
                return self._predict_random_forest(features)
            else:
                return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
    
    def _predict_lstm(self, features: np.ndarray) -> Dict[str, float]:
        """Make prediction using LSTM model."""
        try:
            if self.scaler is None:
                return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Reshape for LSTM (add sequence dimension)
            features_reshaped = features_scaled.reshape(1, 1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_reshaped)[0]
            
            return {
                'buy': float(prediction[1]),
                'sell': float(prediction[2]),
                'hold': float(prediction[0])
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
    
    def _predict_random_forest(self, features: np.ndarray) -> Dict[str, float]:
        """Make prediction using Random Forest model."""
        try:
            if self.scaler is None:
                return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            return {
                'hold': float(probabilities[0]),
                'buy': float(probabilities[1]),
                'sell': float(probabilities[2])
            }
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest prediction: {e}")
            return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
    
    def get_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Get trading signal from the AI model, with caching.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            # Use the latest timestamp as cache key
            cache_key = df.index[-1]
            if cache_key in self._signal_cache:
                self.logger.debug(f"Returning cached AI signal for {cache_key}")
                return self._signal_cache[cache_key]

            # Get prediction from model
            predictions = self.predict(df)
            
            # Get signal based on highest probability
            best_signal = max(predictions, key=predictions.get)
            confidence = predictions[best_signal]
            
            if confidence < self.confidence_threshold:
                signal = 'hold'
            else:
                signal = best_signal
            
            # Cache the result
            result = (signal, confidence)
            self._signal_cache[cache_key] = result

            # Optional: Limit cache size to avoid memory issues
            if len(self._signal_cache) > 500:
                # Remove the oldest item
                oldest_key = next(iter(self._signal_cache))
                del self._signal_cache[oldest_key]

            return result
            
        except Exception as e:
            self.logger.error(f"Error getting AI signal: {e}")
            return 'hold', 0.0

    def save_model(self):
        """Save the trained model and scaler to disk."""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info("AI model and scaler saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving AI model: {e}")
            raise 
"""
Machine Learning Predictor for Cache Optimization
Uses various ML models to predict future cache accesses and optimize performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Optional ML libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class AccessPattern:
    """Represents access pattern features for ML training"""
    
    def __init__(self, key: str, timestamp: float, operation: str):
        self.key = key
        self.timestamp = timestamp
        self.operation = operation
        self.hour_of_day = int((timestamp % 86400) / 3600)
        self.day_of_week = int((timestamp / 86400) % 7)


class FeatureExtractor:
    """Extracts features from access patterns for ML models"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.access_history = deque(maxlen=1000)
        self.key_stats = defaultdict(lambda: {
            'count': 0,
            'last_access': 0,
            'recency_score': 0,
            'frequency_score': 0
        })
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def update_stats(self, key: str, timestamp: float):
        """Update statistics for a key"""
        stats = self.key_stats[key]
        stats['count'] += 1
        stats['last_access'] = timestamp
        stats['recency_score'] = 1.0 / (timestamp - stats['last_access'] + 1)
        stats['frequency_score'] = stats['count'] / len(self.access_history)
        
    def extract_features(self, access_patterns: List[AccessPattern]) -> np.ndarray:
        """Extract features from access patterns"""
        features = []
        
        for pattern in access_patterns:
            self.update_stats(pattern.key, pattern.timestamp)
            
            # Time-based features
            feature_row = [
                pattern.hour_of_day,
                pattern.day_of_week,
                pattern.timestamp % 3600,  # minute of hour
            ]
            
            # Access pattern features
            stats = self.key_stats[pattern.key]
            feature_row.extend([
                stats['count'],
                stats['recency_score'],
                stats['frequency_score'],
                len([p for p in self.access_history if p.key == pattern.key]),
            ])
            
            # Sequential features (last N accesses)
            recent_keys = [p.key for p in list(self.access_history)[-self.window_size:]]
            for i in range(self.window_size):
                if i < len(recent_keys):
                    feature_row.append(hash(recent_keys[i]) % 1000)
                else:
                    feature_row.append(0)
            
            features.append(feature_row)
            self.access_history.append(pattern)
        
        return np.array(features)
    
    def prepare_training_data(self, access_patterns: List[AccessPattern]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        features = self.extract_features(access_patterns)
        
        # Create labels (next access prediction)
        labels = []
        for i in range(len(access_patterns) - 1):
            labels.append(access_patterns[i + 1].key)
        labels.append(access_patterns[-1].key)  # Last item predicts itself
        
        # Encode labels
        try:
            encoded_labels = self.label_encoder.fit_transform(labels)
        except:
            encoded_labels = np.array([hash(label) % 1000 for label in labels])
        
        return features, encoded_labels


class MarkovPredictor:
    """Markov chain-based predictor for next access"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.access_history = deque(maxlen=order)
        
    def train(self, access_patterns: List[AccessPattern]):
        """Train Markov model on access patterns"""
        for pattern in access_patterns:
            if len(self.access_history) == self.order:
                state = tuple(self.access_history)
                next_key = pattern.key
                self.transition_matrix[state][next_key] += 1
                self.state_counts[state] += 1
            
            self.access_history.append(pattern.key)
    
    def predict_next(self, recent_keys: List[str]) -> List[Tuple[str, float]]:
        """Predict next keys with probabilities"""
        if len(recent_keys) < self.order:
            return []
        
        state = tuple(recent_keys[-self.order:])
        if state not in self.transition_matrix:
            return []
        
        predictions = []
        total_count = self.state_counts[state]
        
        for next_key, count in self.transition_matrix[state].items():
            probability = count / total_count
            predictions.append((next_key, probability))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)


class TimeSeriesPredictor:
    """Time series predictor for access patterns"""
    
    def __init__(self):
        self.models = {}
        self.key_timeseries = defaultdict(list)
        
    def train(self, access_patterns: List[AccessPattern]):
        """Train time series models for each key"""
        # Group by key and create time series
        for pattern in access_patterns:
            self.key_timeseries[pattern.key].append(pattern.timestamp)
        
        # Train ARIMA models for frequently accessed keys
        for key, timestamps in self.key_timeseries.items():
            if len(timestamps) > 10 and HAS_STATSMODELS:
                try:
                    # Create hourly access counts
                    df = pd.DataFrame({'timestamp': timestamps})
                    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor('H')
                    hourly_counts = df.groupby('hour').size()
                    
                    if len(hourly_counts) > 5:
                        model = ARIMA(hourly_counts, order=(1, 1, 1))
                        fitted_model = model.fit()
                        self.models[key] = fitted_model
                except:
                    pass
    
    def predict_access_probability(self, key: str, horizon: int = 24) -> float:
        """Predict access probability for a key"""
        if key not in self.models:
            return 0.0
        
        try:
            forecast = self.models[key].forecast(steps=horizon)
            return min(1.0, max(0.0, forecast.mean() / 10))
        except:
            return 0.0


class RFPredictor:
    """Random Forest predictor for cache optimization"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
    def train(self, access_patterns: List[AccessPattern]):
        """Train Random Forest model"""
        features, labels = self.feature_extractor.prepare_training_data(access_patterns)
        
        if len(features) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"RF Predictor Accuracy: {accuracy:.3f}")
    
    def predict_next_keys(self, recent_patterns: List[AccessPattern], top_k: int = 5) -> List[str]:
        """Predict next keys to be accessed"""
        if not self.is_trained:
            return []
        
        features = self.feature_extractor.extract_features(recent_patterns)
        if len(features) == 0:
            return []
        
        # Get prediction probabilities
        try:
            probabilities = self.model.predict_proba(features[-1:])
            top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
            
            # Convert indices back to keys
            predicted_keys = []
            for idx in top_indices:
                if idx < len(self.feature_extractor.label_encoder.classes_):
                    key = self.feature_extractor.label_encoder.inverse_transform([idx])[0]
                    predicted_keys.append(key)
            
            return predicted_keys
        except:
            return []


class LSTMPredictor:
    """LSTM neural network predictor for sequential access patterns"""
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_sequences(self, access_patterns: List[AccessPattern]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        keys = [pattern.key for pattern in access_patterns]
        encoded_keys = self.label_encoder.fit_transform(keys)
        
        X, y = [], []
        for i in range(len(encoded_keys) - self.sequence_length):
            X.append(encoded_keys[i:i + self.sequence_length])
            y.append(encoded_keys[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, access_patterns: List[AccessPattern]):
        """Train LSTM model"""
        if not HAS_TENSORFLOW:
            print("TensorFlow not available, skipping LSTM training")
            return
        
        X, y = self.prepare_sequences(access_patterns)
        if len(X) == 0:
            return
        
        # Reshape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='sparse_categorical_crossentropy', 
                          metrics=['accuracy'])
        
        # Train model
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        self.is_trained = True
        print("LSTM Predictor trained successfully")
    
    def predict_next_keys(self, recent_keys: List[str], top_k: int = 5) -> List[str]:
        """Predict next keys using LSTM"""
        if not self.is_trained or not self.model:
            return []
        
        if len(recent_keys) < self.sequence_length:
            return []
        
        try:
            # Prepare input sequence
            sequence = recent_keys[-self.sequence_length:]
            encoded_sequence = self.label_encoder.transform(sequence)
            X = encoded_sequence.reshape(1, self.sequence_length, 1)
            
            # Get predictions
            predictions = self.model.predict(X, verbose=0)[0]
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            # Convert back to keys
            predicted_keys = []
            for idx in top_indices:
                if idx < len(self.label_encoder.classes_):
                    key = self.label_encoder.inverse_transform([idx])[0]
                    predicted_keys.append(key)
            
            return predicted_keys
        except:
            return []


class EnsemblePredictor:
    """Ensemble predictor combining multiple ML models"""
    
    def __init__(self):
        self.predictors = {
            'markov': MarkovPredictor(order=2),
            'rf': RFPredictor(),
            'timeseries': TimeSeriesPredictor()
        }
        
        if HAS_TENSORFLOW:
            self.predictors['lstm'] = LSTMPredictor()
        
        self.weights = {name: 1.0 for name in self.predictors.keys()}
        self.is_trained = False
    
    def train(self, access_patterns: List[AccessPattern]):
        """Train all predictors"""
        print("Training ensemble predictors...")
        
        for name, predictor in self.predictors.items():
            print(f"Training {name} predictor...")
            try:
                predictor.train(access_patterns)
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        self.is_trained = True
        print("Ensemble training completed")
    
    def predict_next_keys(self, recent_patterns: List[AccessPattern], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next keys using ensemble voting"""
        if not self.is_trained:
            return []
        
        predictions = defaultdict(float)
        recent_keys = [p.key for p in recent_patterns]
        
        # Markov predictions
        if 'markov' in self.predictors:
            markov_preds = self.predictors['markov'].predict_next(recent_keys)
            for key, prob in markov_preds[:top_k]:
                predictions[key] += prob * self.weights['markov']
        
        # Random Forest predictions
        if 'rf' in self.predictors:
            rf_preds = self.predictors['rf'].predict_next_keys(recent_patterns, top_k)
            for i, key in enumerate(rf_preds):
                predictions[key] += (1.0 - i/top_k) * self.weights['rf']
        
        # LSTM predictions
        if 'lstm' in self.predictors and len(recent_keys) > 0:
            lstm_preds = self.predictors['lstm'].predict_next_keys(recent_keys, top_k)
            for i, key in enumerate(lstm_preds):
                predictions[key] += (1.0 - i/top_k) * self.weights['lstm']
        
        # Time series predictions
        if 'timeseries' in self.predictors:
            for key in set(recent_keys):
                prob = self.predictors['timeseries'].predict_access_probability(key)
                predictions[key] += prob * self.weights['timeseries']
        
        # Sort by prediction score
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]
    
    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump(self.predictors, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.predictors = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class OptimalStrategyPredictor:
    """Predicts optimal caching strategy based on workload characteristics"""
    
    def __init__(self):
        self.strategy_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.size_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def extract_workload_features(self, access_patterns: List[AccessPattern]) -> np.ndarray:
        """Extract workload characteristics"""
        if not access_patterns:
            return np.array([])
        
        keys = [p.key for p in access_patterns]
        timestamps = [p.timestamp for p in access_patterns]
        
        # Basic statistics
        unique_keys = len(set(keys))
        total_accesses = len(keys)
        
        # Temporal features
        time_span = max(timestamps) - min(timestamps)
        access_rate = total_accesses / (time_span + 1)
        
        # Access pattern features
        key_counts = defaultdict(int)
        for key in keys:
            key_counts[key] += 1
        
        # Zipf-like distribution measure
        sorted_counts = sorted(key_counts.values(), reverse=True)
        if len(sorted_counts) > 1:
            zipf_measure = sorted_counts[0] / sorted_counts[1] if sorted_counts[1] > 0 else 1.0
        else:
            zipf_measure = 1.0
        
        # Locality measures
        repeated_accesses = sum(1 for i in range(1, len(keys)) if keys[i] in keys[max(0, i-10):i])
        temporal_locality = repeated_accesses / total_accesses if total_accesses > 0 else 0
        
        features = [
            unique_keys,
            total_accesses,
            access_rate,
            zipf_measure,
            temporal_locality,
            unique_keys / total_accesses if total_accesses > 0 else 0,  # diversity
            len([c for c in key_counts.values() if c == 1]) / unique_keys if unique_keys > 0 else 0,  # cold items ratio
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_with_synthetic_data(self):
        """Train with synthetic workload data"""
        from workload_simulator import WorkloadGenerator, WorkloadConfig, WorkloadType
        
        print("Generating synthetic training data for strategy prediction...")
        
        training_data = []
        strategy_labels = []
        size_labels = []
        
        # Generate various workloads and their optimal strategies
        configs = [
            (WorkloadType.SEQUENTIAL, 'FIFO', 50),
            (WorkloadType.RANDOM, 'LRU', 100),
            (WorkloadType.ZIPF, 'LFU', 80),
            (WorkloadType.TEMPORAL_LOCALITY, 'LRU', 60),
            (WorkloadType.SPATIAL_LOCALITY, 'LRU', 70),
            (WorkloadType.MIXED, 'LRU', 90),
        ]
        
        for workload_type, optimal_strategy, optimal_size in configs:
            for _ in range(10):  # Generate multiple samples
                config = WorkloadConfig(
                    workload_type=workload_type,
                    num_requests=500,
                    key_range=200,
                    zipf_parameter=np.random.uniform(0.8, 1.5),
                    locality_factor=np.random.uniform(0.6, 0.9)
                )
                
                generator = WorkloadGenerator(config)
                workload = generator.generate_workload()
                
                # Convert to access patterns
                patterns = [AccessPattern(key, i, 'GET') for i, key in enumerate(workload)]
                features = self.extract_workload_features(patterns)
                
                if features.size > 0:
                    training_data.append(features.flatten())
                    strategy_labels.append(optimal_strategy)
                    size_labels.append(optimal_size)
        
        if training_data:
            X = np.array(training_data)
            y_strategy = np.array(strategy_labels)
            y_size = np.array(size_labels)
            
            # Train models
            self.strategy_model.fit(X, y_strategy)
            self.size_model.fit(X, y_size)
            self.is_trained = True
            
            print("Strategy prediction models trained successfully")
    
    def predict_optimal_strategy(self, access_patterns: List[AccessPattern]) -> Tuple[str, int]:
        """Predict optimal strategy and cache size"""
        if not self.is_trained:
            return 'LRU', 100
        
        features = self.extract_workload_features(access_patterns)
        if features.size == 0:
            return 'LRU', 100
        
        try:
            strategy = self.strategy_model.predict(features)[0]
            size = int(self.size_model.predict(features)[0])
            return strategy, max(10, min(1000, size))
        except:
            return 'LRU', 100


if __name__ == "__main__":
    # Test the predictors
    print("Testing ML Predictors...")
    
    # Generate sample access patterns
    sample_patterns = []
    keys = [f"key_{i}" for i in range(20)]
    
    for i in range(100):
        # Create some temporal locality
        if i > 0 and np.random.random() < 0.3:
            key = sample_patterns[-1].key
        else:
            key = np.random.choice(keys)
        
        pattern = AccessPattern(key, time.time() + i, 'GET')
        sample_patterns.append(pattern)
    
    # Test ensemble predictor
    predictor = EnsemblePredictor()
    predictor.train(sample_patterns)
    
    predictions = predictor.predict_next_keys(sample_patterns[-10:], top_k=5)
    print(f"Predicted next keys: {predictions}")
    
    # Test strategy predictor
    strategy_predictor = OptimalStrategyPredictor()
    strategy_predictor.train_with_synthetic_data()
    
    optimal_strategy, optimal_size = strategy_predictor.predict_optimal_strategy(sample_patterns)
    print(f"Optimal strategy: {optimal_strategy}, size: {optimal_size}")
    
    print("ML Predictors test completed!")

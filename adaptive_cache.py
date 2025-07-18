"""
Adaptive Cache with ML-based Optimizations
"""

from lru_cache_simulator import LRUCache, LFUCache, FIFOCache
from predictor import EnsemblePredictor, OptimalStrategyPredictor, AccessPattern
import time
from typing import List, Any


class AdaptiveCache:
    """
    Adaptive Cache that dynamically optimizes its behavior based on predicted access patterns
    using integrated ML models.
    """
    def __init__(self, initial_capacity: int, strategy: str = 'LRU'):
        self.current_strategy = strategy
        self.capacity = initial_capacity
        self.cache = self._create_cache(self.capacity, self.current_strategy)
        self.predictor = EnsemblePredictor()
        self.strategy_predictor = OptimalStrategyPredictor()
        self.access_patterns: List[AccessPattern] = []

    def _create_cache(self, capacity: int, strategy: str):
        """Create a cache with specified strategy and capacity"""
        if strategy == 'LRU':
            return LRUCache(capacity)
        elif strategy == 'LFU':
            return LFUCache(capacity)
        elif strategy == 'FIFO':
            return FIFOCache(capacity)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def adapt_strategy(self):
        """Adapt cache strategy based on workload analysis"""
        if len(self.access_patterns) >= 100:
            # Train and predict new strategy/settings
            self.strategy_predictor.train_with_synthetic_data()
            predicted_strategy, predicted_size = self.strategy_predictor.predict_optimal_strategy(self.access_patterns)
            
            # Adapt if different
            if predicted_strategy != self.current_strategy or predicted_size != self.capacity:
                print(f"Adapting cache strategy to {predicted_strategy} with size {predicted_size}")
                self.capacity = predicted_size
                self.current_strategy = predicted_strategy
                self.cache = self._create_cache(self.capacity, self.current_strategy)
                
                # Retrain ML models with the latest access patterns
                self.predictor.train(self.access_patterns)

    def get(self, key: Any) -> Any:
        """Retrieve item from cache or auto-fetch based on ML prediction"""
        start_time = time.time()
        value = self.cache.get(key)
        self.access_patterns.append(AccessPattern(key, start_time, 'GET'))

        # Trigger adaptation routinely
        if len(self.access_patterns) % 50 == 0:
            self.adapt_strategy()

        return value

    def put(self, key: Any, value: Any) -> None:
        """Insert item into cache"""
        start_time = time.time()
        self.cache.put(key, value)
        self.access_patterns.append(AccessPattern(key, start_time, 'PUT'))

    def print_stats(self) -> None:
        """Print the current cache statistics"""
        self.cache.print_stats()


# Example usage
if __name__ == "__main__":
    adaptive_cache = AdaptiveCache(initial_capacity=50)

    # Simulate workload
    keys = [f"page_{i}" for i in range(60)]

    for _ in range(200):
        key = keys[int(time.time()) % len(keys)]  # Some repetitive access pattern
        if adaptive_cache.get(key) is None:
            adaptive_cache.put(key, f"Value for {key}")

    adaptive_cache.print_stats()

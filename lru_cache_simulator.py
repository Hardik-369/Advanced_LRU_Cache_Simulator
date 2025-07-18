"""
Advanced LRU Cache Simulator
A comprehensive caching system that simulates real-world caching behavior,
analyzes performance, and compares different eviction strategies.
"""

import time
import json
import csv
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np


class CacheStrategy(Enum):
    """Enumeration of supported cache eviction strategies"""
    LRU = "Least Recently Used"
    LFU = "Least Frequently Used"
    FIFO = "First In First Out"


@dataclass
class CacheStats:
    """Statistics tracking for cache performance"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_accesses: int = 0
    access_times: List[float] = field(default_factory=list)
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage"""
        if self.total_accesses == 0:
            return 0.0
        return (self.hits / self.total_accesses) * 100
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate as percentage"""
        return 100 - self.hit_rate
    
    @property
    def average_access_time(self) -> float:
        """Calculate average access time in milliseconds"""
        if not self.access_times:
            return 0.0
        return sum(self.access_times) / len(self.access_times)


class Node:
    """Doubly linked list node for LRU cache"""
    def __init__(self, key: Any = None, value: Any = None):
        self.key = key
        self.value = value
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None
        self.frequency = 1  # For LFU strategy
        self.timestamp = time.time()  # For FIFO strategy


class LRUCache:
    """
    Advanced LRU Cache implementation using doubly linked list + hash map
    Supports O(1) get and put operations
    """
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")
        
        self.capacity = capacity
        self.cache: Dict[Any, Node] = {}
        self.stats = CacheStats()
        
        # Create dummy head and tail nodes
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Track access patterns for analysis
        self.access_log: List[Tuple[str, Any, float]] = []  # (operation, key, timestamp)
        self.cache_timeline: List[Tuple[float, List[Any]]] = []  # (timestamp, keys_in_cache)
    
    def _add_to_head(self, node: Node) -> None:
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove an existing node from the linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node: Node) -> None:
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _pop_tail(self) -> Node:
        """Remove and return the last node (least recently used)"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value by key in O(1) time"""
        start_time = time.time()
        
        if key in self.cache:
            # Move to head (mark as recently used)
            node = self.cache[key]
            self._move_to_head(node)
            node.frequency += 1  # Update frequency for LFU
            
            self.stats.hits += 1
            self.stats.total_accesses += 1
            
            access_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats.access_times.append(access_time)
            
            self._log_access("GET_HIT", key)
            return node.value
        
        # Cache miss
        self.stats.misses += 1
        self.stats.total_accesses += 1
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._log_access("GET_MISS", key)
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put key-value pair in O(1) time"""
        start_time = time.time()
        
        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            node.frequency += 1
            self._move_to_head(node)
            self._log_access("PUT_UPDATE", key)
        else:
            # Add new key
            new_node = Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove least recently used node
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.stats.evictions += 1
                self._log_access("EVICT", tail.key)
            
            self.cache[key] = new_node
            self._add_to_head(new_node)
            self._log_access("PUT_NEW", key)
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._record_cache_state()
    
    def _log_access(self, operation: str, key: Any) -> None:
        """Log cache access for analysis"""
        self.access_log.append((operation, key, time.time()))
    
    def _record_cache_state(self) -> None:
        """Record current cache state for timeline visualization"""
        current_keys = list(self.cache.keys())
        self.cache_timeline.append((time.time(), current_keys.copy()))
    
    def size(self) -> int:
        """Return current cache size"""
        return len(self.cache)
    
    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return len(self.cache) == 0
    
    def is_full(self) -> bool:
        """Check if cache is at capacity"""
        return len(self.cache) >= self.capacity
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.stats = CacheStats()
        self.access_log.clear()
        self.cache_timeline.clear()
    
    def keys(self) -> List[Any]:
        """Return all keys in cache (most recent first)"""
        keys = []
        current = self.head.next
        while current != self.tail:
            keys.append(current.key)
            current = current.next
        return keys
    
    def report_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        return {
            "capacity": self.capacity,
            "current_size": self.size(),
            "total_accesses": self.stats.total_accesses,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "hit_rate_percent": round(self.stats.hit_rate, 2),
            "miss_rate_percent": round(self.stats.miss_rate, 2),
            "average_access_time_ms": round(self.stats.average_access_time, 4),
            "memory_efficiency": round((self.size() / self.capacity) * 100, 2)
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics report"""
        stats = self.report_stats()
        print("\n" + "="*50)
        print("CACHE PERFORMANCE REPORT")
        print("="*50)
        print(f"Strategy: LRU (Least Recently Used)")
        print(f"Capacity: {stats['capacity']}")
        print(f"Current Size: {stats['current_size']}")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"Miss Rate: {stats['miss_rate_percent']}%")
        print(f"Average Access Time: {stats['average_access_time_ms']} ms")
        print(f"Memory Efficiency: {stats['memory_efficiency']}%")
        print("="*50)


class LFUCache:
    """
    Least Frequently Used Cache implementation
    Evicts the least frequently used items
    """
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")
        
        self.capacity = capacity
        self.cache: Dict[Any, Any] = {}
        self.frequencies: Dict[Any, int] = {}
        self.stats = CacheStats()
        self.access_log: List[Tuple[str, Any, float]] = []
        self.cache_timeline: List[Tuple[float, List[Any]]] = []
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value by key"""
        start_time = time.time()
        
        if key in self.cache:
            self.frequencies[key] += 1
            self.stats.hits += 1
            self.stats.total_accesses += 1
            
            access_time = (time.time() - start_time) * 1000
            self.stats.access_times.append(access_time)
            
            self._log_access("GET_HIT", key)
            return self.cache[key]
        
        self.stats.misses += 1
        self.stats.total_accesses += 1
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._log_access("GET_MISS", key)
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put key-value pair"""
        start_time = time.time()
        
        if key in self.cache:
            self.cache[key] = value
            self.frequencies[key] += 1
            self._log_access("PUT_UPDATE", key)
        else:
            if len(self.cache) >= self.capacity:
                # Find and remove least frequently used item
                lfu_key = min(self.frequencies, key=self.frequencies.get)
                del self.cache[lfu_key]
                del self.frequencies[lfu_key]
                self.stats.evictions += 1
                self._log_access("EVICT", lfu_key)
            
            self.cache[key] = value
            self.frequencies[key] = 1
            self._log_access("PUT_NEW", key)
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._record_cache_state()
    
    def _log_access(self, operation: str, key: Any) -> None:
        """Log cache access for analysis"""
        self.access_log.append((operation, key, time.time()))
    
    def _record_cache_state(self) -> None:
        """Record current cache state for timeline visualization"""
        current_keys = list(self.cache.keys())
        self.cache_timeline.append((time.time(), current_keys.copy()))
    
    def size(self) -> int:
        return len(self.cache)
    
    def clear(self) -> None:
        self.cache.clear()
        self.frequencies.clear()
        self.stats = CacheStats()
        self.access_log.clear()
        self.cache_timeline.clear()
    
    def keys(self) -> List[Any]:
        return list(self.cache.keys())
    
    def report_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        return {
            "capacity": self.capacity,
            "current_size": self.size(),
            "total_accesses": self.stats.total_accesses,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "hit_rate_percent": round(self.stats.hit_rate, 2),
            "miss_rate_percent": round(self.stats.miss_rate, 2),
            "average_access_time_ms": round(self.stats.average_access_time, 4),
            "memory_efficiency": round((self.size() / self.capacity) * 100, 2)
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics report"""
        stats = self.report_stats()
        print("\n" + "="*50)
        print("CACHE PERFORMANCE REPORT")
        print("="*50)
        print(f"Strategy: LFU (Least Frequently Used)")
        print(f"Capacity: {stats['capacity']}")
        print(f"Current Size: {stats['current_size']}")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"Miss Rate: {stats['miss_rate_percent']}%")
        print(f"Average Access Time: {stats['average_access_time_ms']} ms")
        print(f"Memory Efficiency: {stats['memory_efficiency']}%")
        print("="*50)


class FIFOCache:
    """
    First In First Out Cache implementation
    Evicts the oldest items first
    """
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")
        
        self.capacity = capacity
        self.cache = OrderedDict()
        self.stats = CacheStats()
        self.access_log: List[Tuple[str, Any, float]] = []
        self.cache_timeline: List[Tuple[float, List[Any]]] = []
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value by key"""
        start_time = time.time()
        
        if key in self.cache:
            self.stats.hits += 1
            self.stats.total_accesses += 1
            
            access_time = (time.time() - start_time) * 1000
            self.stats.access_times.append(access_time)
            
            self._log_access("GET_HIT", key)
            return self.cache[key]
        
        self.stats.misses += 1
        self.stats.total_accesses += 1
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._log_access("GET_MISS", key)
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put key-value pair"""
        start_time = time.time()
        
        if key in self.cache:
            # Update existing key (don't change order in FIFO)
            self.cache[key] = value
            self._log_access("PUT_UPDATE", key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove oldest item (first in)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.evictions += 1
                self._log_access("EVICT", oldest_key)
            
            self.cache[key] = value
            self._log_access("PUT_NEW", key)
        
        access_time = (time.time() - start_time) * 1000
        self.stats.access_times.append(access_time)
        
        self._record_cache_state()
    
    def _log_access(self, operation: str, key: Any) -> None:
        """Log cache access for analysis"""
        self.access_log.append((operation, key, time.time()))
    
    def _record_cache_state(self) -> None:
        """Record current cache state for timeline visualization"""
        current_keys = list(self.cache.keys())
        self.cache_timeline.append((time.time(), current_keys.copy()))
    
    def size(self) -> int:
        return len(self.cache)
    
    def clear(self) -> None:
        self.cache.clear()
        self.stats = CacheStats()
        self.access_log.clear()
        self.cache_timeline.clear()
    
    def keys(self) -> List[Any]:
        return list(self.cache.keys())
    
    def report_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        return {
            "capacity": self.capacity,
            "current_size": self.size(),
            "total_accesses": self.stats.total_accesses,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "hit_rate_percent": round(self.stats.hit_rate, 2),
            "miss_rate_percent": round(self.stats.miss_rate, 2),
            "average_access_time_ms": round(self.stats.average_access_time, 4),
            "memory_efficiency": round((self.size() / self.capacity) * 100, 2)
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics report"""
        stats = self.report_stats()
        print("\n" + "="*50)
        print("CACHE PERFORMANCE REPORT")
        print("="*50)
        print(f"Strategy: FIFO (First In First Out)")
        print(f"Capacity: {stats['capacity']}")
        print(f"Current Size: {stats['current_size']}")
        print(f"Total Accesses: {stats['total_accesses']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"Miss Rate: {stats['miss_rate_percent']}%")
        print(f"Average Access Time: {stats['average_access_time_ms']} ms")
        print(f"Memory Efficiency: {stats['memory_efficiency']}%")
        print("="*50)


if __name__ == "__main__":
    # Basic demonstration
    print("Advanced LRU Cache Simulator")
    print("="*40)
    
    # Create cache
    cache = LRUCache(capacity=3)
    
    # Test basic operations
    print("\nTesting basic operations...")
    cache.put("A", "Value A")
    cache.put("B", "Value B")
    cache.put("C", "Value C")
    
    print(f"Cache keys: {cache.keys()}")
    print(f"Get A: {cache.get('A')}")
    print(f"Cache keys after access: {cache.keys()}")
    
    # This should evict B (least recently used)
    cache.put("D", "Value D")
    print(f"Cache keys after adding D: {cache.keys()}")
    
    # Print statistics
    cache.print_stats()

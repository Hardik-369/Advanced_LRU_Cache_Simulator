"""
Redis Benchmark Module
Compares our cache simulator performance with Redis caching system
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import subprocess
import os
import threading
import queue

# Redis client
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from lru_cache_simulator import LRUCache, LFUCache, FIFOCache
from workload_simulator import WorkloadGenerator, WorkloadConfig, WorkloadType


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    system_name: str
    total_operations: int
    hit_rate: float
    miss_rate: float
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    errors: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'system_name': self.system_name,
            'total_operations': self.total_operations,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'memory_usage_mb': self.memory_usage_mb,
            'errors': self.errors
        }


class RedisManager:
    """Manages Redis server instance for benchmarking"""
    
    def __init__(self, port: int = 6379):
        self.port = port
        self.redis_process = None
        self.client = None
        
    def start_redis_server(self) -> bool:
        """Start Redis server if not already running"""
        if not HAS_REDIS:
            print("Redis client not available. Install with: pip install redis")
            return False
        
        try:
            # Try to connect to existing Redis instance
            self.client = redis.Redis(host='localhost', port=self.port, decode_responses=True)
            self.client.ping()
            print(f"Connected to existing Redis server on port {self.port}")
            return True
        except:
            print(f"No Redis server found on port {self.port}")
            
            # Try to start Redis server (Windows)
            if os.name == 'nt':
                print("Please start Redis server manually on Windows")
                print("Download Redis from: https://github.com/microsoftarchive/redis/releases")
                return False
            
            # Try to start Redis server (Unix-like systems)
            try:
                print("Attempting to start Redis server...")
                self.redis_process = subprocess.Popen(
                    ['redis-server', '--port', str(self.port), '--daemonize', 'no'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait a bit for server to start
                time.sleep(2)
                
                # Try to connect
                self.client = redis.Redis(host='localhost', port=self.port, decode_responses=True)
                self.client.ping()
                print(f"Started Redis server on port {self.port}")
                return True
                
            except Exception as e:
                print(f"Failed to start Redis server: {e}")
                return False
    
    def stop_redis_server(self):
        """Stop Redis server"""
        if self.redis_process:
            self.redis_process.terminate()
            self.redis_process.wait()
            print("Redis server stopped")
    
    def get_client(self):
        """Get Redis client"""
        return self.client


class RedisBenchmark:
    """Benchmark Redis performance"""
    
    def __init__(self, redis_client, max_memory_mb: int = 100):
        self.client = redis_client
        self.max_memory_mb = max_memory_mb
        self.hits = 0
        self.misses = 0
        self.latencies = []
        self.errors = 0
        
        # Configure Redis for LRU eviction
        try:
            self.client.config_set('maxmemory', f'{max_memory_mb}mb')
            self.client.config_set('maxmemory-policy', 'allkeys-lru')
        except:
            pass
    
    def benchmark_workload(self, workload: List[str], value_size: int = 100) -> BenchmarkResult:
        """Benchmark Redis with given workload"""
        self.hits = 0
        self.misses = 0
        self.latencies = []
        self.errors = 0
        
        # Clear Redis
        self.client.flushdb()
        
        # Generate a sample value
        sample_value = 'x' * value_size
        
        start_time = time.time()
        
        for key in workload:
            # Try to get from cache
            op_start = time.time()
            try:
                result = self.client.get(key)
                op_end = time.time()
                
                if result is not None:
                    self.hits += 1
                else:
                    self.misses += 1
                    # Cache miss - set the value
                    self.client.set(key, sample_value)
                
                self.latencies.append((op_end - op_start) * 1000)  # Convert to ms
                
            except Exception as e:
                self.errors += 1
                print(f"Redis error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_ops = len(workload)
        hit_rate = (self.hits / total_ops) * 100 if total_ops > 0 else 0
        miss_rate = (self.misses / total_ops) * 100 if total_ops > 0 else 0
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        throughput = total_ops / total_time if total_time > 0 else 0
        
        # Estimate memory usage
        try:
            memory_info = self.client.info('memory')
            memory_used_mb = memory_info['used_memory'] / (1024 * 1024)
        except:
            memory_used_mb = 0
        
        return BenchmarkResult(
            system_name='Redis',
            total_operations=total_ops,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_used_mb,
            errors=self.errors
        )


class SimulatorBenchmark:
    """Benchmark our cache simulator"""
    
    def __init__(self, cache_type: str = 'LRU', capacity: int = 1000):
        self.cache_type = cache_type
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.latencies = []
        self.errors = 0
        
    def benchmark_workload(self, workload: List[str], value_size: int = 100) -> BenchmarkResult:
        """Benchmark our simulator with given workload"""
        # Create cache
        if self.cache_type == 'LRU':
            cache = LRUCache(self.capacity)
        elif self.cache_type == 'LFU':
            cache = LFUCache(self.capacity)
        elif self.cache_type == 'FIFO':
            cache = FIFOCache(self.capacity)
        else:
            raise ValueError(f"Unknown cache type: {self.cache_type}")
        
        self.hits = 0
        self.misses = 0
        self.latencies = []
        self.errors = 0
        
        # Generate a sample value
        sample_value = 'x' * value_size
        
        start_time = time.time()
        
        for key in workload:
            # Try to get from cache
            op_start = time.time()
            try:
                result = cache.get(key)
                op_end = time.time()
                
                if result is not None:
                    self.hits += 1
                else:
                    self.misses += 1
                    # Cache miss - set the value
                    cache.put(key, sample_value)
                
                self.latencies.append((op_end - op_start) * 1000)  # Convert to ms
                
            except Exception as e:
                self.errors += 1
                print(f"Simulator error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_ops = len(workload)
        hit_rate = (self.hits / total_ops) * 100 if total_ops > 0 else 0
        miss_rate = (self.misses / total_ops) * 100 if total_ops > 0 else 0
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        throughput = total_ops / total_time if total_time > 0 else 0
        
        # Estimate memory usage (approximate)
        memory_used_mb = (cache.size() * value_size) / (1024 * 1024)
        
        return BenchmarkResult(
            system_name=f'Simulator-{self.cache_type}',
            total_operations=total_ops,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_used_mb,
            errors=self.errors
        )


class BenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def __init__(self):
        self.redis_manager = RedisManager()
        self.results = []
        
    def run_comprehensive_benchmark(self, workload_configs: List[WorkloadConfig], 
                                   cache_capacity: int = 1000) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing Redis with simulator"""
        print("Starting comprehensive benchmark...")
        
        # Try to start Redis
        redis_available = self.redis_manager.start_redis_server()
        
        all_results = {}
        
        for config in workload_configs:
            print(f"\\nBenchmarking {config.workload_type.value} workload...")
            
            # Generate workload
            generator = WorkloadGenerator(config)
            workload = generator.generate_workload()
            
            workload_name = f"{config.workload_type.value}_{config.num_requests}req"
            workload_results = []
            
            # Benchmark our simulators
            for cache_type in ['LRU', 'LFU', 'FIFO']:
                print(f"  Testing {cache_type} simulator...")
                simulator = SimulatorBenchmark(cache_type, cache_capacity)
                result = simulator.benchmark_workload(workload)
                workload_results.append(result)
            
            # Benchmark Redis if available
            if redis_available:
                print("  Testing Redis...")
                redis_benchmark = RedisBenchmark(
                    self.redis_manager.get_client(),
                    max_memory_mb=cache_capacity // 10  # Rough conversion
                )
                redis_result = redis_benchmark.benchmark_workload(workload)
                workload_results.append(redis_result)
            
            all_results[workload_name] = {
                'config': {
                    'workload_type': config.workload_type.value,
                    'num_requests': config.num_requests,
                    'key_range': config.key_range,
                    'cache_capacity': cache_capacity
                },
                'results': [r.to_dict() for r in workload_results]
            }
        
        return all_results
    
    def export_results(self, results: Dict[str, Any], filename: str):
        """Export benchmark results to JSON"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {filename}")
    
    def print_comparison_report(self, results: Dict[str, Any]):
        """Print formatted comparison report"""
        print("\\n" + "="*80)
        print("CACHE SYSTEM BENCHMARK COMPARISON")
        print("="*80)
        
        for workload_name, workload_data in results.items():
            print(f"\\n{workload_name.upper()}:")
            print("-" * 60)
            
            # Print header
            print(f"{'System':<20} {'Hit Rate':<10} {'Latency(ms)':<12} {'Throughput':<15} {'Memory(MB)':<12}")
            print("-" * 60)
            
            for result in workload_data['results']:
                print(f"{result['system_name']:<20} "
                      f"{result['hit_rate']:<10.2f} "
                      f"{result['avg_latency_ms']:<12.4f} "
                      f"{result['throughput_ops_per_sec']:<15.2f} "
                      f"{result['memory_usage_mb']:<12.2f}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.redis_manager.stop_redis_server()


def simulate_real_world_scenario():
    """Simulate a real-world caching scenario"""
    print("Simulating real-world web server cache scenario...")
    
    # Create a mixed workload representing web traffic
    config = WorkloadConfig(
        workload_type=WorkloadType.ZIPF,
        num_requests=5000,
        key_range=1000,
        zipf_parameter=1.1  # Realistic web traffic distribution
    )
    
    suite = BenchmarkSuite()
    results = suite.run_comprehensive_benchmark([config], cache_capacity=200)
    
    suite.print_comparison_report(results)
    suite.export_results(results, 'redis_vs_simulator_benchmark.json')
    
    suite.cleanup()
    
    return results


if __name__ == "__main__":
    # Run simulation without Redis dependency
    print("Redis Benchmark Module Test")
    print("="*50)
    
    # Test simulator benchmark
    simulator = SimulatorBenchmark('LRU', 100)
    test_workload = [f"key_{i}" for i in range(200)] * 2  # Some repetition
    
    print("Testing simulator benchmark...")
    result = simulator.benchmark_workload(test_workload)
    
    print(f"System: {result.system_name}")
    print(f"Hit Rate: {result.hit_rate:.2f}%")
    print(f"Avg Latency: {result.avg_latency_ms:.4f} ms")
    print(f"Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
    print(f"Memory Usage: {result.memory_usage_mb:.2f} MB")
    
    # Test full benchmark if Redis is available
    if HAS_REDIS:
        print("\\nTesting full benchmark with Redis...")
        try:
            simulate_real_world_scenario()
        except Exception as e:
            print(f"Redis benchmark failed: {e}")
    else:
        print("\\nRedis not available - skipping Redis benchmark")
        print("Install Redis and redis-py to enable Redis benchmarking")
    
    print("\\nBenchmark test completed!")

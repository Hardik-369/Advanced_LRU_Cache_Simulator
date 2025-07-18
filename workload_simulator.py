"""
Workload Simulator and Strategy Comparison Engine
Generates various access patterns and compares cache strategies
"""

import random
import math
import time
from typing import List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import csv
from lru_cache_simulator import LRUCache, LFUCache, FIFOCache, CacheStrategy


class WorkloadType(Enum):
    """Types of workload patterns"""
    SEQUENTIAL = "Sequential Access"
    RANDOM = "Random Access"
    ZIPF = "Zipf Distribution"
    TEMPORAL_LOCALITY = "Temporal Locality"
    SPATIAL_LOCALITY = "Spatial Locality"
    MIXED = "Mixed Pattern"


@dataclass
class WorkloadConfig:
    """Configuration for workload generation"""
    workload_type: WorkloadType
    num_requests: int
    key_range: int
    zipf_parameter: float = 1.0  # For Zipf distribution
    locality_factor: float = 0.8  # For temporal/spatial locality
    burst_size: int = 10  # For temporal locality bursts
    custom_pattern: List[Any] = None


class WorkloadGenerator:
    """Generates various types of access patterns for cache simulation"""
    
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.random = random.Random(42)  # Fixed seed for reproducibility
    
    def generate_workload(self) -> List[Any]:
        """Generate access pattern based on configuration"""
        if self.config.workload_type == WorkloadType.SEQUENTIAL:
            return self._generate_sequential()
        elif self.config.workload_type == WorkloadType.RANDOM:
            return self._generate_random()
        elif self.config.workload_type == WorkloadType.ZIPF:
            return self._generate_zipf()
        elif self.config.workload_type == WorkloadType.TEMPORAL_LOCALITY:
            return self._generate_temporal_locality()
        elif self.config.workload_type == WorkloadType.SPATIAL_LOCALITY:
            return self._generate_spatial_locality()
        elif self.config.workload_type == WorkloadType.MIXED:
            return self._generate_mixed()
        else:
            raise ValueError(f"Unknown workload type: {self.config.workload_type}")
    
    def _generate_sequential(self) -> List[Any]:
        """Generate sequential access pattern"""
        pattern = []
        for i in range(self.config.num_requests):
            key = f"key_{i % self.config.key_range}"
            pattern.append(key)
        return pattern
    
    def _generate_random(self) -> List[Any]:
        """Generate random access pattern"""
        pattern = []
        for _ in range(self.config.num_requests):
            key = f"key_{self.random.randint(0, self.config.key_range - 1)}"
            pattern.append(key)
        return pattern
    
    def _generate_zipf(self) -> List[Any]:
        """Generate Zipf distribution access pattern"""
        # Pre-calculate Zipf probabilities
        zipf_probs = []
        harmonic_sum = sum(1.0 / (i ** self.config.zipf_parameter) 
                          for i in range(1, self.config.key_range + 1))
        
        for i in range(1, self.config.key_range + 1):
            prob = (1.0 / (i ** self.config.zipf_parameter)) / harmonic_sum
            zipf_probs.append(prob)
        
        # Generate access pattern
        pattern = []
        for _ in range(self.config.num_requests):
            # Sample from Zipf distribution
            rand_val = self.random.random()
            cumulative = 0.0
            for i, prob in enumerate(zipf_probs):
                cumulative += prob
                if rand_val <= cumulative:
                    key = f"key_{i}"
                    pattern.append(key)
                    break
        
        return pattern
    
    def _generate_temporal_locality(self) -> List[Any]:
        """Generate pattern with temporal locality (recent items accessed again)"""
        pattern = []
        recent_keys = []
        
        for _ in range(self.config.num_requests):
            if recent_keys and self.random.random() < self.config.locality_factor:
                # Access recent key
                key = self.random.choice(recent_keys)
            else:
                # Access new key
                key = f"key_{self.random.randint(0, self.config.key_range - 1)}"
                recent_keys.append(key)
                
                # Limit recent keys list size
                if len(recent_keys) > self.config.burst_size:
                    recent_keys.pop(0)
            
            pattern.append(key)
        
        return pattern
    
    def _generate_spatial_locality(self) -> List[Any]:
        """Generate pattern with spatial locality (nearby keys accessed together)"""
        pattern = []
        current_base = 0
        
        for _ in range(self.config.num_requests):
            if self.random.random() < self.config.locality_factor:
                # Access nearby key
                offset = self.random.randint(-self.config.burst_size // 2, 
                                           self.config.burst_size // 2)
                key_idx = max(0, min(self.config.key_range - 1, current_base + offset))
            else:
                # Jump to new location
                current_base = self.random.randint(0, self.config.key_range - 1)
                key_idx = current_base
            
            key = f"key_{key_idx}"
            pattern.append(key)
        
        return pattern
    
    def _generate_mixed(self) -> List[Any]:
        """Generate mixed access pattern combining different types"""
        pattern = []
        segment_size = self.config.num_requests // 4
        
        # Sequential segment
        for i in range(segment_size):
            key = f"key_{i % (self.config.key_range // 4)}"
            pattern.append(key)
        
        # Random segment
        for _ in range(segment_size):
            key = f"key_{self.random.randint(0, self.config.key_range - 1)}"
            pattern.append(key)
        
        # Zipf segment
        zipf_config = WorkloadConfig(
            WorkloadType.ZIPF, segment_size, self.config.key_range,
            zipf_parameter=1.2
        )
        zipf_gen = WorkloadGenerator(zipf_config)
        pattern.extend(zipf_gen._generate_zipf())
        
        # Temporal locality segment
        remaining = self.config.num_requests - len(pattern)
        temporal_config = WorkloadConfig(
            WorkloadType.TEMPORAL_LOCALITY, remaining, self.config.key_range,
            locality_factor=0.7
        )
        temporal_gen = WorkloadGenerator(temporal_config)
        pattern.extend(temporal_gen._generate_temporal_locality())
        
        return pattern


class CacheSimulator:
    """Simulates cache behavior with different strategies and workloads"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def simulate_strategy(self, strategy: CacheStrategy, workload: List[Any], 
                         strategy_name: str = None) -> Dict[str, Any]:
        """Simulate cache behavior for a specific strategy and workload"""
        if strategy_name is None:
            strategy_name = strategy.value
        
        # Create cache based on strategy
        if strategy == CacheStrategy.LRU:
            cache = LRUCache(self.capacity)
        elif strategy == CacheStrategy.LFU:
            cache = LFUCache(self.capacity)
        elif strategy == CacheStrategy.FIFO:
            cache = FIFOCache(self.capacity)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Simulate workload
        start_time = time.time()
        
        for key in workload:
            # Try to get from cache
            value = cache.get(key)
            
            # If not found, simulate fetching from source and put in cache
            if value is None:
                # Simulate fetch latency
                time.sleep(0.0001)  # 0.1ms simulated fetch time
                value = f"value_for_{key}"
                cache.put(key, value)
        
        simulation_time = time.time() - start_time
        
        # Collect results
        stats = cache.report_stats()
        stats['simulation_time_seconds'] = round(simulation_time, 4)
        stats['strategy'] = strategy_name
        stats['workload_size'] = len(workload)
        
        return stats
    
    def compare_strategies(self, workload: List[Any], 
                          strategies: List[CacheStrategy] = None) -> Dict[str, Any]:
        """Compare multiple cache strategies on the same workload"""
        if strategies is None:
            strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO]
        
        results = {}
        
        for strategy in strategies:
            strategy_name = strategy.value
            print(f"Simulating {strategy_name}...")
            
            result = self.simulate_strategy(strategy, workload, strategy_name)
            results[strategy_name] = result
        
        return results
    
    def run_comprehensive_analysis(self, workload_configs: List[WorkloadConfig],
                                 strategies: List[CacheStrategy] = None) -> Dict[str, Any]:
        """Run comprehensive analysis across multiple workloads and strategies"""
        if strategies is None:
            strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO]
        
        comprehensive_results = {}
        
        for config in workload_configs:
            print(f"\nAnalyzing {config.workload_type.value} workload...")
            
            # Generate workload
            generator = WorkloadGenerator(config)
            workload = generator.generate_workload()
            
            # Compare strategies
            workload_results = self.compare_strategies(workload, strategies)
            
            workload_name = f"{config.workload_type.value}_{config.num_requests}req"
            comprehensive_results[workload_name] = {
                'config': {
                    'workload_type': config.workload_type.value,
                    'num_requests': config.num_requests,
                    'key_range': config.key_range,
                    'zipf_parameter': config.zipf_parameter,
                    'locality_factor': config.locality_factor
                },
                'results': workload_results
            }
        
        return comprehensive_results


class ReportGenerator:
    """Generates detailed reports and exports data"""
    
    @staticmethod
    def print_comparison_report(results: Dict[str, Any]) -> None:
        """Print formatted comparison report"""
        print("\n" + "="*80)
        print("CACHE STRATEGY COMPARISON REPORT")
        print("="*80)
        
        # Find best strategy for each metric
        strategies = list(results.keys())
        metrics = ['hit_rate_percent', 'average_access_time_ms', 'memory_efficiency']
        
        best_performers = {}
        for metric in metrics:
            best_strategy = max(strategies, key=lambda s: results[s][metric])
            best_performers[metric] = best_strategy
        
        # Print summary table
        print(f"{'Strategy':<20} {'Hit Rate':<12} {'Avg Time':<12} {'Memory Eff':<12} {'Evictions':<12}")
        print("-" * 80)
        
        for strategy, stats in results.items():
            print(f"{strategy:<20} "
                  f"{stats['hit_rate_percent']:<12.2f} "
                  f"{stats['average_access_time_ms']:<12.4f} "
                  f"{stats['memory_efficiency']:<12.2f} "
                  f"{stats['evictions']:<12}")
        
        print("\nBest Performers:")
        print(f"• Highest Hit Rate: {best_performers['hit_rate_percent']} "
              f"({results[best_performers['hit_rate_percent']]['hit_rate_percent']:.2f}%)")
        print(f"• Lowest Access Time: {best_performers['average_access_time_ms']} "
              f"({results[best_performers['average_access_time_ms']]['average_access_time_ms']:.4f} ms)")
        print(f"• Best Memory Efficiency: {best_performers['memory_efficiency']} "
              f"({results[best_performers['memory_efficiency']]['memory_efficiency']:.2f}%)")
        
        print("="*80)
    
    @staticmethod
    def export_to_csv(results: Dict[str, Any], filename: str) -> None:
        """Export results to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            if not results:
                return
            
            # Get all field names from first result
            first_result = next(iter(results.values()))
            if isinstance(first_result, dict) and 'results' in first_result:
                # Comprehensive analysis format
                fieldnames = ['workload_type', 'strategy', 'capacity', 'hit_rate_percent',
                            'miss_rate_percent', 'average_access_time_ms', 'memory_efficiency',
                            'total_accesses', 'hits', 'misses', 'evictions', 'simulation_time_seconds']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for workload_name, workload_data in results.items():
                    workload_type = workload_data['config']['workload_type']
                    for strategy, stats in workload_data['results'].items():
                        row = {
                            'workload_type': workload_type,
                            'strategy': strategy,
                            **{k: v for k, v in stats.items() if k in fieldnames}
                        }
                        writer.writerow(row)
            else:
                # Simple comparison format
                fieldnames = list(first_result.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for strategy, stats in results.items():
                    row = {'strategy': strategy, **stats}
                    writer.writerow(row)
        
        print(f"Results exported to {filename}")
    
    @staticmethod
    def export_to_json(results: Dict[str, Any], filename: str) -> None:
        """Export results to JSON file"""
        with open(filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2, default=str)
        
        print(f"Results exported to {filename}")


def fetch_from_source(key: Any) -> str:
    """Simulate fetching data from source (database, API, etc.)"""
    # Simulate some processing time
    time.sleep(0.001)  # 1ms simulated fetch time
    return f"value_for_{key}"


def load_trace_from_file(filename: str) -> List[Any]:
    """Load access trace from file"""
    try:
        with open(filename, 'r') as f:
            trace = []
            for line in f:
                line = line.strip()
                if line:
                    trace.append(line)
            return trace
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Using sample trace.")
        return [f"key_{i}" for i in range(100)]


if __name__ == "__main__":
    # Example usage
    print("Cache Strategy Comparison Demo")
    print("="*50)
    
    # Create simulator
    simulator = CacheSimulator(capacity=50)
    
    # Define workload configurations
    workload_configs = [
        WorkloadConfig(WorkloadType.SEQUENTIAL, 1000, 200),
        WorkloadConfig(WorkloadType.RANDOM, 1000, 200),
        WorkloadConfig(WorkloadType.ZIPF, 1000, 200, zipf_parameter=1.2),
        WorkloadConfig(WorkloadType.TEMPORAL_LOCALITY, 1000, 200, locality_factor=0.8),
    ]
    
    # Run comprehensive analysis
    results = simulator.run_comprehensive_analysis(workload_configs)
    
    # Generate reports
    report_gen = ReportGenerator()
    
    # Print individual workload reports
    for workload_name, workload_data in results.items():
        print(f"\n{workload_name.upper()} RESULTS:")
        report_gen.print_comparison_report(workload_data['results'])
    
    # Export results
    report_gen.export_to_csv(results, "cache_analysis_results.csv")
    report_gen.export_to_json(results, "cache_analysis_results.json")
    
    print("\nAnalysis complete!")

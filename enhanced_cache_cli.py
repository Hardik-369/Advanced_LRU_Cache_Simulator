#!/usr/bin/env python3
"""
Enhanced Cache Simulator CLI with ML Predictions and Redis Benchmarking
"""

import argparse
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import all modules
from cache_simulator_cli import CacheSimulatorCLI
from predictor import EnsemblePredictor, OptimalStrategyPredictor, AccessPattern
from adaptive_cache import AdaptiveCache
from redis_benchmark import BenchmarkSuite, SimulatorBenchmark
from workload_simulator import WorkloadGenerator, WorkloadConfig, WorkloadType
from log_parser import parse_access_log, log_analysis_report
from cache_visualizer import CacheVisualizer


class EnhancedCacheSimulatorCLI:
    """Enhanced CLI with ML and benchmarking capabilities"""
    
    def __init__(self):
        self.parser = self.create_enhanced_parser()
        self.visualizer = CacheVisualizer()
        
    def create_enhanced_parser(self) -> argparse.ArgumentParser:
        """Create enhanced argument parser"""
        parser = argparse.ArgumentParser(
            description="Enhanced Cache Simulator with ML and Redis Benchmarking",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Enhanced Features:
  --ml-predict         Enable ML-based access prediction
  --adaptive           Use adaptive caching with ML optimization
  --benchmark-redis    Compare performance with Redis
  --train-ml          Train ML models on workload data
  --log-analysis      Analyze real-world log files
  --hybrid-cache      Use hybrid caching strategies
  --research-mode     Generate comprehensive research data
  
Examples:
  # ML-enhanced adaptive caching
  python enhanced_cache_cli.py --capacity 100 --adaptive --ml-predict
  
  # Benchmark against Redis
  python enhanced_cache_cli.py --benchmark-redis --capacity 100 --workload zipf
  
  # Train ML models and generate predictions
  python enhanced_cache_cli.py --train-ml --workload temporal --requests 2000
  
  # Real-world log analysis
  python enhanced_cache_cli.py --log-analysis access.log --capacity 200
  
  # Research mode with comprehensive analysis
  python enhanced_cache_cli.py --research-mode --export-all
            """
        )
        
        # Core parameters
        parser.add_argument('--capacity', type=int, default=100,
                          help='Cache capacity (default: 100)')
        
        # Enhanced simulation modes
        parser.add_argument('--ml-predict', action='store_true',
                          help='Enable ML-based access prediction')
        parser.add_argument('--adaptive', action='store_true',
                          help='Use adaptive caching with ML optimization')
        parser.add_argument('--benchmark-redis', action='store_true',
                          help='Compare performance with Redis')
        parser.add_argument('--train-ml', action='store_true',
                          help='Train ML models on workload data')
        parser.add_argument('--log-analysis', type=str,
                          help='Analyze real-world log file')
        parser.add_argument('--hybrid-cache', action='store_true',
                          help='Use hybrid caching strategies')
        parser.add_argument('--research-mode', action='store_true',
                          help='Generate comprehensive research data')
        
        # Workload parameters
        parser.add_argument('--workload', 
                          choices=['sequential', 'random', 'zipf', 'temporal', 'spatial', 'mixed'],
                          default='zipf',
                          help='Workload type for simulation')
        parser.add_argument('--requests', type=int, default=1000,
                          help='Number of requests to simulate')
        parser.add_argument('--key-range', type=int, default=500,
                          help='Range of keys to use')
        
        # ML parameters
        parser.add_argument('--ml-model', 
                          choices=['ensemble', 'markov', 'rf', 'lstm'],
                          default='ensemble',
                          help='ML model to use for predictions')
        parser.add_argument('--prediction-window', type=int, default=10,
                          help='Number of future accesses to predict')
        parser.add_argument('--adaptation-threshold', type=int, default=100,
                          help='Number of accesses before strategy adaptation')
        
        # Export options
        parser.add_argument('--export-all', action='store_true',
                          help='Export all results and models')
        parser.add_argument('--output-dir', type=str, default='research_output',
                          help='Output directory for research data')
        
        # Performance options
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Quiet mode')
        
        return parser
    
    def run_ml_prediction_demo(self, args) -> None:
        """Demonstrate ML-based access prediction"""
        print("🧠 ML-Enhanced Cache Prediction Demo")
        print("=" * 60)
        
        # Generate workload
        config = WorkloadConfig(
            workload_type=WorkloadType.TEMPORAL_LOCALITY,
            num_requests=args.requests,
            key_range=args.key_range,
            locality_factor=0.8
        )
        
        generator = WorkloadGenerator(config)
        workload = generator.generate_workload()
        
        # Convert to access patterns
        access_patterns = []
        for i, key in enumerate(workload[:100]):  # Train on first 100 accesses
            pattern = AccessPattern(key, time.time() + i, 'GET')
            access_patterns.append(pattern)
        
        # Train ML predictor
        print("Training ML predictor...")
        predictor = EnsemblePredictor()
        predictor.train(access_patterns)
        
        # Make predictions
        print(f"Making predictions for next {args.prediction_window} accesses...")
        predictions = predictor.predict_next_keys(
            access_patterns[-10:], 
            top_k=args.prediction_window
        )
        
        print("\\nPredictions:")
        for i, (key, confidence) in enumerate(predictions):
            print(f"  {i+1}. {key} (confidence: {confidence:.3f})")
        
        # Test predictions against actual accesses
        actual_next = workload[100:100+args.prediction_window]
        predicted_keys = [p[0] for p in predictions]
        
        hits = len(set(predicted_keys) & set(actual_next))
        accuracy = hits / len(actual_next) if actual_next else 0
        
        print(f"\\nPrediction Accuracy: {accuracy:.2%}")
        print(f"Predicted: {predicted_keys}")
        print(f"Actual:    {actual_next}")
    
    def run_adaptive_cache_demo(self, args) -> None:
        """Demonstrate adaptive caching with ML optimization"""
        print("🔄 Adaptive Cache with ML Optimization Demo")
        print("=" * 60)
        
        # Create adaptive cache
        adaptive_cache = AdaptiveCache(
            initial_capacity=args.capacity,
            strategy='LRU'
        )
        
        # Generate diverse workloads to trigger adaptation
        workload_types = [
            WorkloadType.SEQUENTIAL,
            WorkloadType.ZIPF,
            WorkloadType.TEMPORAL_LOCALITY,
            WorkloadType.RANDOM
        ]
        
        for workload_type in workload_types:
            print(f"\\nRunning {workload_type.value} workload...")
            
            config = WorkloadConfig(
                workload_type=workload_type,
                num_requests=args.requests // 4,
                key_range=args.key_range
            )
            
            generator = WorkloadGenerator(config)
            workload = generator.generate_workload()
            
            # Process workload
            for key in workload:
                value = adaptive_cache.get(key)
                if value is None:
                    adaptive_cache.put(key, f"value_for_{key}")
        
        print("\\nFinal Adaptive Cache Statistics:")
        adaptive_cache.print_stats()
    
    def run_redis_benchmark(self, args) -> None:
        """Run Redis benchmark comparison"""
        print("⚡ Redis vs Simulator Benchmark")
        print("=" * 60)
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        
        # Define workload
        config = WorkloadConfig(
            workload_type=WorkloadType.ZIPF if args.workload == 'zipf' else WorkloadType.RANDOM,
            num_requests=args.requests,
            key_range=args.key_range,
            zipf_parameter=1.2
        )
        
        # Run benchmark
        results = suite.run_comprehensive_benchmark([config], args.capacity)
        
        # Display results
        suite.print_comparison_report(results)
        
        # Export results
        if args.export_all:
            output_file = os.path.join(args.output_dir, 'redis_benchmark_results.json')
            os.makedirs(args.output_dir, exist_ok=True)
            suite.export_results(results, output_file)
        
        suite.cleanup()
    
    def run_log_analysis(self, args) -> None:
        """Analyze real-world log files"""
        print(f"📊 Real-World Log Analysis: {args.log_analysis}")
        print("=" * 60)
        
        # Parse log file
        access_keys = parse_access_log(args.log_analysis)
        
        if not access_keys:
            print("No access keys found in log file")
            return
        
        print(f"Parsed {len(access_keys)} access records")
        
        # Generate analysis report
        report = log_analysis_report(access_keys)
        
        print("\\nLog Analysis Report:")
        print(f"Total Requests: {report['total_requests']}")
        print(f"Unique Keys: {report['unique_keys']}")
        print(f"Most Frequent Key: {report['most_frequent_key']} ({report['most_frequent_key_count']} times)")
        
        # Test cache strategies on real data
        print("\\nTesting cache strategies on real data...")
        
        strategies = ['LRU', 'LFU', 'FIFO']
        results = {}
        
        for strategy in strategies:
            simulator = SimulatorBenchmark(strategy, args.capacity)
            result = simulator.benchmark_workload(access_keys)
            results[strategy] = result
            
            print(f"{strategy}: {result.hit_rate:.2f}% hit rate, "
                  f"{result.avg_latency_ms:.4f}ms avg latency")
        
        # Export results
        if args.export_all:
            output_file = os.path.join(args.output_dir, 'log_analysis_results.json')
            os.makedirs(args.output_dir, exist_ok=True)
            
            export_data = {
                'log_analysis': report,
                'cache_performance': {k: v.to_dict() for k, v in results.items()}
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"\\nResults exported to {output_file}")
    
    def run_research_mode(self, args) -> None:
        """Generate comprehensive research data"""
        print("🔬 Research Mode - Comprehensive Analysis")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 1. Train ML models
        print("\\n1. Training ML Models...")
        self.run_ml_prediction_demo(args)
        
        # 2. Test adaptive caching
        print("\\n2. Testing Adaptive Caching...")
        self.run_adaptive_cache_demo(args)
        
        # 3. Benchmark against Redis (if available)
        try:
            print("\\n3. Benchmarking Against Redis...")
            self.run_redis_benchmark(args)
        except Exception as e:
            print(f"Redis benchmark failed: {e}")
        
        # 4. Generate comprehensive comparison
        print("\\n4. Generating Comprehensive Comparison...")
        
        # Multiple workload types
        workload_configs = [
            WorkloadConfig(WorkloadType.SEQUENTIAL, args.requests, args.key_range),
            WorkloadConfig(WorkloadType.RANDOM, args.requests, args.key_range),
            WorkloadConfig(WorkloadType.ZIPF, args.requests, args.key_range, zipf_parameter=1.2),
            WorkloadConfig(WorkloadType.TEMPORAL_LOCALITY, args.requests, args.key_range),
            WorkloadConfig(WorkloadType.MIXED, args.requests, args.key_range),
        ]
        
        # Run comprehensive analysis
        from workload_simulator import CacheSimulator, ReportGenerator
        
        simulator = CacheSimulator(capacity=args.capacity)
        results = simulator.run_comprehensive_analysis(workload_configs)
        
        # Export comprehensive results
        output_file = os.path.join(args.output_dir, 'comprehensive_research_results.json')
        ReportGenerator.export_to_json(results, output_file)
        
        # Generate visualizations
        print("\\n5. Generating Visualizations...")
        try:
            self.visualizer.plot_workload_comparison(results)
            self.visualizer.export_visualization_report(results, args.output_dir)
        except Exception as e:
            print(f"Visualization generation failed: {e}")
        
        print(f"\\n✅ Research data generated in {args.output_dir}/")
    
    def run(self, args=None):
        """Main entry point for enhanced CLI"""
        if args is None:
            args = self.parser.parse_args()
        
        # Print header
        if not args.quiet:
            print("🚀 Enhanced Cache Simulator with ML & Benchmarking")
            print(f"⚙️  Capacity: {args.capacity}")
            print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            if args.ml_predict:
                self.run_ml_prediction_demo(args)
            elif args.adaptive:
                self.run_adaptive_cache_demo(args)
            elif args.benchmark_redis:
                self.run_redis_benchmark(args)
            elif args.train_ml:
                self.run_ml_prediction_demo(args)
            elif args.log_analysis:
                self.run_log_analysis(args)
            elif args.research_mode:
                self.run_research_mode(args)
            else:
                # Fall back to standard CLI
                standard_cli = CacheSimulatorCLI()
                standard_cli.run(args)
                return
            
            # Show execution time
            if not args.quiet:
                elapsed = time.time() - start_time
                print(f"\\n✅ Enhanced simulation completed in {elapsed:.2f} seconds")
                
        except KeyboardInterrupt:
            print("\\n⚠️  Simulation interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for enhanced CLI"""
    cli = EnhancedCacheSimulatorCLI()
    cli.run()


if __name__ == "__main__":
    main()

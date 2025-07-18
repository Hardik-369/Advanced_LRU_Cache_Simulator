#!/usr/bin/env python3
"""
Advanced LRU Cache Simulator - Command Line Interface
Interactive CLI for running cache simulations and comparisons
"""

import argparse
import sys
import os
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Import our modules
from lru_cache_simulator import LRUCache, LFUCache, FIFOCache, CacheStrategy
from workload_simulator import (
    WorkloadGenerator, WorkloadConfig, WorkloadType, CacheSimulator, 
    ReportGenerator, load_trace_from_file
)
from cache_visualizer import CacheVisualizer


class CacheSimulatorCLI:
    """Command-line interface for cache simulation"""
    
    def __init__(self):
        self.parser = self.create_parser()
        self.simulator = None
        self.visualizer = CacheVisualizer()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Advanced LRU Cache Simulator",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic cache test
  python cache_simulator_cli.py --capacity 100 --demo
  
  # Run comprehensive analysis
  python cache_simulator_cli.py --capacity 50 --comprehensive
  
  # Compare strategies on custom workload
  python cache_simulator_cli.py --capacity 100 --workload random --requests 2000 --key-range 500
  
  # Load trace from file
  python cache_simulator_cli.py --capacity 100 --trace-file access.log
  
  # Export results to CSV
  python cache_simulator_cli.py --capacity 100 --comprehensive --export-csv results.csv
  
  # Generate visualizations
  python cache_simulator_cli.py --capacity 100 --demo --visualize
            """
        )
        
        # Core simulation parameters
        parser.add_argument('--capacity', type=int, default=100,
                          help='Cache capacity (default: 100)')
        
        # Workload options
        workload_group = parser.add_mutually_exclusive_group()
        workload_group.add_argument('--demo', action='store_true',
                                   help='Run basic demonstration')
        workload_group.add_argument('--comprehensive', action='store_true',
                                   help='Run comprehensive analysis with multiple workloads')
        workload_group.add_argument('--workload', 
                                   choices=['sequential', 'random', 'zipf', 'temporal', 'spatial', 'mixed'],
                                   help='Workload type for simulation')
        workload_group.add_argument('--trace-file', type=str,
                                   help='Load access trace from file')
        workload_group.add_argument('--interactive', action='store_true',
                                   help='Interactive mode for custom testing')
        
        # Workload parameters
        parser.add_argument('--requests', type=int, default=1000,
                          help='Number of requests to simulate (default: 1000)')
        parser.add_argument('--key-range', type=int, default=200,
                          help='Range of keys to use (default: 200)')
        parser.add_argument('--zipf-param', type=float, default=1.2,
                          help='Zipf distribution parameter (default: 1.2)')
        parser.add_argument('--locality-factor', type=float, default=0.8,
                          help='Locality factor for temporal/spatial workloads (default: 0.8)')
        
        # Strategy selection
        parser.add_argument('--strategies', nargs='+', 
                          choices=['lru', 'lfu', 'fifo', 'all'],
                          default=['all'],
                          help='Cache strategies to test (default: all)')
        
        # Output options
        parser.add_argument('--export-csv', type=str,
                          help='Export results to CSV file')
        parser.add_argument('--export-json', type=str,
                          help='Export results to JSON file')
        parser.add_argument('--output-dir', type=str, default='.',
                          help='Output directory for exports (default: current directory)')
        
        # Visualization options
        parser.add_argument('--visualize', action='store_true',
                          help='Generate visualizations')
        parser.add_argument('--save-plots', action='store_true',
                          help='Save visualization plots to files')
        parser.add_argument('--live-monitor', action='store_true',
                          help='Show live monitoring dashboard (demo mode)')
        
        # Performance options
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Quiet mode (minimal output)')
        
        return parser
    
    def parse_strategies(self, strategy_args: List[str]) -> List[CacheStrategy]:
        """Parse strategy arguments"""
        if 'all' in strategy_args:
            return [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO]
        
        strategy_map = {
            'lru': CacheStrategy.LRU,
            'lfu': CacheStrategy.LFU,
            'fifo': CacheStrategy.FIFO
        }
        
        return [strategy_map[s] for s in strategy_args if s in strategy_map]
    
    def run_demo(self, args) -> None:
        """Run basic demonstration"""
        print("🚀 Running Cache Simulator Demo")
        print("=" * 50)
        
        # Create LRU cache for demo
        cache = LRUCache(capacity=args.capacity)
        
        # Simple operations
        print("\n📝 Basic Operations Demo:")
        demo_ops = [
            ("put", "page1", "content1"),
            ("put", "page2", "content2"),
            ("put", "page3", "content3"),
            ("get", "page1", None),
            ("put", "page4", "content4"),  # Should evict page2
            ("get", "page2", None),  # Should be miss
            ("get", "page3", None),
            ("put", "page5", "content5"),
        ]
        
        for op, key, value in demo_ops:
            if op == "put":
                cache.put(key, value)
                print(f"  PUT {key} -> Cache keys: {cache.keys()}")
            else:
                result = cache.get(key)
                status = "HIT" if result else "MISS"
                print(f"  GET {key} -> {status}, Cache keys: {cache.keys()}")
        
        print(f"\n📊 Final Statistics:")
        cache.print_stats()
        
        if args.visualize:
            print("\n📈 Generating visualizations...")
            try:
                self.visualizer.plot_cache_timeline(cache)
                self.visualizer.plot_access_pattern_heatmap(cache)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        if args.live_monitor:
            print("\n🔴 Live Monitor Demo:")
            self.visualizer.create_live_cache_monitor(cache)
    
    def run_comprehensive(self, args) -> None:
        """Run comprehensive analysis"""
        print("🔬 Running Comprehensive Cache Analysis")
        print("=" * 50)
        
        self.simulator = CacheSimulator(capacity=args.capacity)
        strategies = self.parse_strategies(args.strategies)
        
        # Define workload configurations
        workload_configs = [
            WorkloadConfig(WorkloadType.SEQUENTIAL, args.requests, args.key_range),
            WorkloadConfig(WorkloadType.RANDOM, args.requests, args.key_range),
            WorkloadConfig(WorkloadType.ZIPF, args.requests, args.key_range, 
                         zipf_parameter=args.zipf_param),
            WorkloadConfig(WorkloadType.TEMPORAL_LOCALITY, args.requests, args.key_range,
                         locality_factor=args.locality_factor),
            WorkloadConfig(WorkloadType.SPATIAL_LOCALITY, args.requests, args.key_range,
                         locality_factor=args.locality_factor),
            WorkloadConfig(WorkloadType.MIXED, args.requests, args.key_range),
        ]
        
        print(f"📋 Testing {len(strategies)} strategies on {len(workload_configs)} workloads...")
        
        # Run analysis
        results = self.simulator.run_comprehensive_analysis(workload_configs, strategies)
        
        # Generate reports
        report_gen = ReportGenerator()
        
        if not args.quiet:
            for workload_name, workload_data in results.items():
                print(f"\n📊 {workload_name.upper()} RESULTS:")
                report_gen.print_comparison_report(workload_data['results'])
        
        # Export results
        if args.export_csv:
            report_gen.export_to_csv(results, args.export_csv)
        
        if args.export_json:
            report_gen.export_to_json(results, args.export_json)
        
        # Visualizations
        if args.visualize:
            print("\n📈 Generating comprehensive visualizations...")
            self.visualizer.plot_workload_comparison(results)
            
            if args.save_plots:
                self.visualizer.export_visualization_report(results, args.output_dir)
        
        return results
    
    def run_custom_workload(self, args) -> None:
        """Run simulation with custom workload"""
        print(f"🎯 Running {args.workload.upper()} Workload Simulation")
        print("=" * 50)
        
        self.simulator = CacheSimulator(capacity=args.capacity)
        strategies = self.parse_strategies(args.strategies)
        
        # Map workload types
        workload_map = {
            'sequential': WorkloadType.SEQUENTIAL,
            'random': WorkloadType.RANDOM,
            'zipf': WorkloadType.ZIPF,
            'temporal': WorkloadType.TEMPORAL_LOCALITY,
            'spatial': WorkloadType.SPATIAL_LOCALITY,
            'mixed': WorkloadType.MIXED
        }
        
        # Create workload config
        config = WorkloadConfig(
            workload_type=workload_map[args.workload],
            num_requests=args.requests,
            key_range=args.key_range,
            zipf_parameter=args.zipf_param,
            locality_factor=args.locality_factor
        )
        
        # Generate workload
        generator = WorkloadGenerator(config)
        workload = generator.generate_workload()
        
        print(f"📋 Generated {len(workload)} requests with {len(set(workload))} unique keys")
        
        # Run simulation
        results = self.simulator.compare_strategies(workload, strategies)
        
        # Generate report
        report_gen = ReportGenerator()
        if not args.quiet:
            report_gen.print_comparison_report(results)
        
        # Export results
        if args.export_csv:
            report_gen.export_to_csv(results, args.export_csv)
        
        if args.export_json:
            report_gen.export_to_json(results, args.export_json)
        
        # Visualizations
        if args.visualize:
            print("\n📈 Generating visualizations...")
            self.visualizer.plot_hit_miss_ratio(results)
            self.visualizer.plot_performance_metrics(results)
            
            if args.save_plots:
                self.visualizer.export_visualization_report(results, args.output_dir)
    
    def run_trace_file(self, args) -> None:
        """Run simulation with trace file"""
        print(f"📁 Loading trace from {args.trace_file}")
        print("=" * 50)
        
        # Load trace
        trace = load_trace_from_file(args.trace_file)
        
        if not trace:
            print("❌ Could not load trace file")
            return
        
        print(f"📋 Loaded {len(trace)} requests with {len(set(trace))} unique keys")
        
        # Run simulation
        self.simulator = CacheSimulator(capacity=args.capacity)
        strategies = self.parse_strategies(args.strategies)
        
        results = self.simulator.compare_strategies(trace, strategies)
        
        # Generate report
        report_gen = ReportGenerator()
        if not args.quiet:
            report_gen.print_comparison_report(results)
        
        # Export results
        if args.export_csv:
            report_gen.export_to_csv(results, args.export_csv)
        
        if args.export_json:
            report_gen.export_to_json(results, args.export_json)
        
        # Visualizations
        if args.visualize:
            print("\n📈 Generating visualizations...")
            self.visualizer.plot_hit_miss_ratio(results)
            self.visualizer.plot_performance_metrics(results)
    
    def run_interactive(self, args) -> None:
        """Run interactive mode"""
        print("🎮 Interactive Cache Simulator")
        print("=" * 50)
        print("Commands: get <key>, put <key> <value>, stats, keys, clear, quit")
        print("Try: put page1 content1, get page1, put page2 content2, etc.")
        print()
        
        # Create cache
        cache = LRUCache(capacity=args.capacity)
        
        while True:
            try:
                command = input("cache> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'get':
                    if len(command) < 2:
                        print("Usage: get <key>")
                        continue
                    key = command[1]
                    value = cache.get(key)
                    if value:
                        print(f"HIT: {key} -> {value}")
                    else:
                        print(f"MISS: {key}")
                elif cmd == 'put':
                    if len(command) < 3:
                        print("Usage: put <key> <value>")
                        continue
                    key = command[1]
                    value = ' '.join(command[2:])
                    cache.put(key, value)
                    print(f"PUT: {key} -> {value}")
                elif cmd == 'stats':
                    cache.print_stats()
                elif cmd == 'keys':
                    print(f"Cache keys: {cache.keys()}")
                elif cmd == 'clear':
                    cache.clear()
                    print("Cache cleared")
                elif cmd == 'help':
                    print("Commands: get <key>, put <key> <value>, stats, keys, clear, quit")
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Show final stats
        if not args.quiet:
            print("\n📊 Final Statistics:")
            cache.print_stats()
    
    def run(self, args=None):
        """Main entry point"""
        if args is None:
            args = self.parser.parse_args()
        
        # Print header
        if not args.quiet:
            print("🏃‍♂️ Advanced LRU Cache Simulator")
            print(f"⚙️  Capacity: {args.capacity}")
            print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            if args.demo:
                self.run_demo(args)
            elif args.comprehensive:
                self.run_comprehensive(args)
            elif args.workload:
                self.run_custom_workload(args)
            elif args.trace_file:
                self.run_trace_file(args)
            elif args.interactive:
                self.run_interactive(args)
            else:
                print("❌ No simulation mode specified. Use --help for options.")
                return
            
            # Show execution time
            if not args.quiet:
                elapsed = time.time() - start_time
                print(f"\n✅ Simulation completed in {elapsed:.2f} seconds")
                
        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for CLI"""
    cli = CacheSimulatorCLI()
    cli.run()


if __name__ == "__main__":
    main()

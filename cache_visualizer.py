"""
Cache Visualization and Plotting Utilities
Provides real-time and static visualization of cache behavior
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import time
from datetime import datetime
from collections import defaultdict
import json

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CacheVisualizer:
    """Real-time and static visualization for cache behavior"""
    
    def __init__(self, cache_capacity: int = 100):
        self.cache_capacity = cache_capacity
        self.fig = None
        self.axes = None
        self.animation_data = []
        
    def plot_hit_miss_ratio(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot hit/miss ratios for different strategies"""
        strategies = list(results.keys())
        hit_rates = [results[s]['hit_rate_percent'] for s in strategies]
        miss_rates = [results[s]['miss_rate_percent'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, hit_rates, width, label='Hit Rate', color='#2E8B57')
        bars2 = ax.bar(x + width/2, miss_rates, width, label='Miss Rate', color='#DC143C')
        
        ax.set_xlabel('Cache Strategy')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Cache Hit/Miss Rates by Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_metrics(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot multiple performance metrics in subplots"""
        strategies = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cache Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Hit Rate
        hit_rates = [results[s]['hit_rate_percent'] for s in strategies]
        axes[0, 0].bar(strategies, hit_rates, color='#2E8B57', alpha=0.8)
        axes[0, 0].set_title('Hit Rate (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(hit_rates):
            axes[0, 0].text(i, v + max(hit_rates)*0.01, f'{v:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Average Access Time
        access_times = [results[s]['average_access_time_ms'] for s in strategies]
        axes[0, 1].bar(strategies, access_times, color='#4169E1', alpha=0.8)
        axes[0, 1].set_title('Average Access Time (ms)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(access_times):
            axes[0, 1].text(i, v + max(access_times)*0.01, f'{v:.3f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Memory Efficiency
        memory_effs = [results[s]['memory_efficiency'] for s in strategies]
        axes[1, 0].bar(strategies, memory_effs, color='#FF6347', alpha=0.8)
        axes[1, 0].set_title('Memory Efficiency (%)')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(memory_effs):
            axes[1, 0].text(i, v + max(memory_effs)*0.01, f'{v:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Evictions
        evictions = [results[s]['evictions'] for s in strategies]
        axes[1, 1].bar(strategies, evictions, color='#FF8C00', alpha=0.8)
        axes[1, 1].set_title('Total Evictions')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(evictions):
            axes[1, 1].text(i, v + max(evictions)*0.01, f'{v}', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_workload_comparison(self, comprehensive_results: Dict[str, Any], 
                               save_path: str = None) -> None:
        """Plot performance across different workload types"""
        # Prepare data for plotting
        workload_types = []
        strategies = None
        data = defaultdict(lambda: defaultdict(list))
        
        for workload_name, workload_data in comprehensive_results.items():
            workload_type = workload_data['config']['workload_type']
            workload_types.append(workload_type)
            
            if strategies is None:
                strategies = list(workload_data['results'].keys())
            
            for strategy, stats in workload_data['results'].items():
                data[strategy]['hit_rate'].append(stats['hit_rate_percent'])
                data[strategy]['access_time'].append(stats['average_access_time_ms'])
                data[strategy]['memory_eff'].append(stats['memory_efficiency'])
        
        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Cache Performance Across Different Workload Types', 
                    fontsize=16, fontweight='bold')
        
        x = np.arange(len(workload_types))
        width = 0.25
        colors = ['#2E8B57', '#4169E1', '#FF6347']
        
        # Hit Rate comparison
        for i, strategy in enumerate(strategies):
            offset = (i - 1) * width
            bars = axes[0].bar(x + offset, data[strategy]['hit_rate'], width, 
                             label=strategy, color=colors[i], alpha=0.8)
        
        axes[0].set_xlabel('Workload Type')
        axes[0].set_ylabel('Hit Rate (%)')
        axes[0].set_title('Hit Rate by Workload Type')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(workload_types, rotation=45)
        axes[0].legend()
        
        # Access Time comparison
        for i, strategy in enumerate(strategies):
            offset = (i - 1) * width
            bars = axes[1].bar(x + offset, data[strategy]['access_time'], width, 
                             label=strategy, color=colors[i], alpha=0.8)
        
        axes[1].set_xlabel('Workload Type')
        axes[1].set_ylabel('Access Time (ms)')
        axes[1].set_title('Access Time by Workload Type')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(workload_types, rotation=45)
        axes[1].legend()
        
        # Memory Efficiency comparison
        for i, strategy in enumerate(strategies):
            offset = (i - 1) * width
            bars = axes[2].bar(x + offset, data[strategy]['memory_eff'], width, 
                             label=strategy, color=colors[i], alpha=0.8)
        
        axes[2].set_xlabel('Workload Type')
        axes[2].set_ylabel('Memory Efficiency (%)')
        axes[2].set_title('Memory Efficiency by Workload Type')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(workload_types, rotation=45)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cache_timeline(self, cache_obj, save_path: str = None) -> None:
        """Plot cache state timeline showing keys over time"""
        if not hasattr(cache_obj, 'cache_timeline') or not cache_obj.cache_timeline:
            print("No timeline data available. Run cache operations first.")
            return
        
        # Extract timeline data
        timestamps = [t[0] for t in cache_obj.cache_timeline]
        cache_states = [t[1] for t in cache_obj.cache_timeline]
        
        # Normalize timestamps to start from 0
        start_time = timestamps[0]
        timestamps = [(t - start_time) * 1000 for t in timestamps]  # Convert to ms
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot cache size over time
        cache_sizes = [len(state) for state in cache_states]
        ax.plot(timestamps, cache_sizes, 'b-', linewidth=2, label='Cache Size')
        ax.axhline(y=cache_obj.capacity, color='r', linestyle='--', 
                  label=f'Capacity ({cache_obj.capacity})')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Cache Size')
        ax.set_title('Cache Size Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add capacity utilization percentage
        ax2 = ax.twinx()
        utilization = [(size / cache_obj.capacity) * 100 for size in cache_sizes]
        ax2.plot(timestamps, utilization, 'g--', alpha=0.7, label='Utilization %')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_access_pattern_heatmap(self, cache_obj, save_path: str = None) -> None:
        """Plot heatmap of access patterns"""
        if not hasattr(cache_obj, 'access_log') or not cache_obj.access_log:
            print("No access log data available.")
            return
        
        # Extract access data
        access_data = defaultdict(int)
        operation_counts = defaultdict(int)
        
        for operation, key, timestamp in cache_obj.access_log:
            access_data[key] += 1
            operation_counts[operation] += 1
        
        # Create heatmap data
        keys = sorted(access_data.keys())
        access_counts = [access_data[key] for key in keys]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Access frequency heatmap
        if len(keys) > 50:
            # Show top 50 most accessed keys
            top_keys = sorted(access_data.items(), key=lambda x: x[1], reverse=True)[:50]
            keys = [k for k, v in top_keys]
            access_counts = [v for k, v in top_keys]
        
        # Create heatmap
        heatmap_data = np.array(access_counts).reshape(1, -1)
        im1 = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Access Frequency Heatmap')
        ax1.set_xlabel('Keys')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(0, len(keys), max(1, len(keys)//10)))
        ax1.set_xticklabels([keys[i] for i in range(0, len(keys), max(1, len(keys)//10))], 
                           rotation=45)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Access Count')
        
        # Operation type pie chart
        operations = list(operation_counts.keys())
        counts = list(operation_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        ax2.pie(counts, labels=operations, autopct='%1.1f%%', colors=colors[:len(operations)])
        ax2.set_title('Cache Operations Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_live_cache_monitor(self, cache_obj, max_history: int = 100) -> None:
        """Create a live monitoring dashboard for cache performance"""
        # This would be used with a real-time cache in production
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Live Cache Performance Monitor', fontsize=16, fontweight='bold')
        
        # Initialize data storage
        self.monitor_data = {
            'timestamps': [],
            'hit_rates': [],
            'cache_sizes': [],
            'response_times': [],
            'operation_counts': {'GET_HIT': 0, 'GET_MISS': 0, 'PUT_NEW': 0, 'PUT_UPDATE': 0, 'EVICT': 0}
        }
        
        def update_monitor(frame):
            # This would be called periodically to update the live dashboard
            current_time = time.time()
            
            # Update data (in real implementation, this would come from cache metrics)
            self.monitor_data['timestamps'].append(current_time)
            self.monitor_data['hit_rates'].append(cache_obj.stats.hit_rate if hasattr(cache_obj, 'stats') else 0)
            self.monitor_data['cache_sizes'].append(cache_obj.size() if hasattr(cache_obj, 'size') else 0)
            self.monitor_data['response_times'].append(
                cache_obj.stats.average_access_time if hasattr(cache_obj, 'stats') else 0
            )
            
            # Keep only recent data
            if len(self.monitor_data['timestamps']) > max_history:
                for key in ['timestamps', 'hit_rates', 'cache_sizes', 'response_times']:
                    self.monitor_data[key] = self.monitor_data[key][-max_history:]
            
            # Clear and update plots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # Hit rate over time
            ax1.plot(self.monitor_data['timestamps'], self.monitor_data['hit_rates'], 'b-')
            ax1.set_title('Hit Rate Over Time')
            ax1.set_ylabel('Hit Rate (%)')
            ax1.grid(True, alpha=0.3)
            
            # Cache size over time
            ax2.plot(self.monitor_data['timestamps'], self.monitor_data['cache_sizes'], 'g-')
            ax2.axhline(y=cache_obj.capacity, color='r', linestyle='--', label='Capacity')
            ax2.set_title('Cache Size Over Time')
            ax2.set_ylabel('Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Response time over time
            ax3.plot(self.monitor_data['timestamps'], self.monitor_data['response_times'], 'r-')
            ax3.set_title('Response Time Over Time')
            ax3.set_ylabel('Time (ms)')
            ax3.grid(True, alpha=0.3)
            
            # Current cache statistics
            if hasattr(cache_obj, 'stats'):
                stats_text = f"""
                Total Accesses: {cache_obj.stats.total_accesses}
                Hits: {cache_obj.stats.hits}
                Misses: {cache_obj.stats.misses}
                Evictions: {cache_obj.stats.evictions}
                Hit Rate: {cache_obj.stats.hit_rate:.2f}%
                Avg Access Time: {cache_obj.stats.average_access_time:.4f} ms
                """
                ax4.text(0.1, 0.7, stats_text, transform=ax4.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax4.set_title('Current Statistics')
            ax4.axis('off')
            
            plt.tight_layout()
        
        # Create animation (for demonstration purposes)
        print("Live monitoring dashboard created (demonstration mode)")
        print("In a real implementation, this would update in real-time")
        
        # Show static version
        update_monitor(0)
        plt.show()
    
    def export_visualization_report(self, results: Dict[str, Any], 
                                  output_dir: str = ".") -> None:
        """Export a comprehensive visualization report"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate different visualizations
        print("Generating comprehensive visualization report...")
        
        # 1. Hit/Miss ratios
        self.plot_hit_miss_ratio(results, 
                                save_path=os.path.join(output_dir, f"hit_miss_ratio_{timestamp}.png"))
        
        # 2. Performance metrics
        self.plot_performance_metrics(results, 
                                    save_path=os.path.join(output_dir, f"performance_metrics_{timestamp}.png"))
        
        # 3. If comprehensive results available
        if isinstance(list(results.values())[0], dict) and 'results' in list(results.values())[0]:
            self.plot_workload_comparison(results, 
                                        save_path=os.path.join(output_dir, f"workload_comparison_{timestamp}.png"))
        
        print(f"Visualization report exported to {output_dir}")


def create_sample_visualization():
    """Create sample visualizations for demonstration"""
    # Sample data for demonstration
    sample_results = {
        'Least Recently Used': {
            'hit_rate_percent': 75.2,
            'miss_rate_percent': 24.8,
            'average_access_time_ms': 0.0234,
            'memory_efficiency': 98.5,
            'evictions': 45
        },
        'Least Frequently Used': {
            'hit_rate_percent': 72.8,
            'miss_rate_percent': 27.2,
            'average_access_time_ms': 0.0198,
            'memory_efficiency': 95.2,
            'evictions': 52
        },
        'First In First Out': {
            'hit_rate_percent': 68.4,
            'miss_rate_percent': 31.6,
            'average_access_time_ms': 0.0256,
            'memory_efficiency': 94.8,
            'evictions': 58
        }
    }
    
    visualizer = CacheVisualizer()
    
    print("Creating sample visualizations...")
    
    # Create visualizations
    visualizer.plot_hit_miss_ratio(sample_results)
    visualizer.plot_performance_metrics(sample_results)
    
    print("Sample visualizations created!")


if __name__ == "__main__":
    create_sample_visualization()

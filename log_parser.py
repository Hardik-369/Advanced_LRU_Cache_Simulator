"""
Log Parser for Real-World Access Traces
"""

import json
from typing import List, Dict
import pandas as pd


def parse_access_log(log_path: str) -> List[str]:
    """
    Parse an access log file to extract key access sequences.
    Assumes each log line is structured as a common access log format.
    """
    keys = []
    
    try:
        with open(log_path, 'r') as log_file:
            for line in log_file:
                # Simplified log parsing for demonstration (you may need to adapt based on actual format)
                parts = line.split(' ')
                if len(parts) > 6:
                    # Extract URL and convert to a key
                    url = parts[6]
                    key = url.split('/')[-1]  # Example: '/product/123' -> '123'
                    if key:
                        keys.append(key)
    except FileNotFoundError:
        print(f"File not found: {log_path}")
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return keys


def extract_key_from_log_entry(log_line: str) -> str:
    """
    Extract key from a single log line based on specific parsing rules.
    """
    try:
        parts = log_line.split(' ')
        if len(parts) > 6:
            url = parts[6]
            return url.split('/')[-1]  # Example: '/product/123' -> '123'
    except Exception:
        pass
    return ""


def log_analysis_report(logs: List[str]) -> Dict[str, any]:
    """
    Generate a report based on access frequency, patterns, and other metrics.
    """
    df = pd.DataFrame(logs, columns=['key'])
    
    # Frequency analysis
    freq = df['key'].value_counts()
    
    # Collect insights
    insights = {
        'total_requests': len(logs),
        'unique_keys': freq.shape[0],
        'most_frequent_key': freq.idxmax() if not freq.empty else None,
        'most_frequent_key_count': freq.max() if not freq.empty else 0,
        'least_frequent_key': freq.idxmin() if not freq.empty else None,
        'key_distribution': freq.to_dict()
    }
    
    return insights


if __name__ == "__main__":
    # Test log parsing with a sample log file
    log_file_path = 'sample_access_log.txt'
    
    print(f"Parsing {log_file_path}...")
    access_keys = parse_access_log(log_file_path)
    if access_keys:
        print("Parsed access keys:", access_keys[:10], "...")
        report = log_analysis_report(access_keys)
        print(json.dumps(report, indent=4))
    else:
        print("No access keys parsed from the log.")

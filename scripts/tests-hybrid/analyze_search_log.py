#!/usr/bin/env python3
"""
Analyze search_process.log to check if all queries returned k results.
Parses "Total Results: X / K (all satisfy tag constraints)" lines.
"""

import re
import sys
from collections import defaultdict

def analyze_log(log_path):
    # Pattern to match "Total Results: X / K (all satisfy tag constraints)"
    total_results_pattern = re.compile(
        r'- Total Results:\s+(\d+)\s+/\s+(\d+)\s+\(all satisfy tag constraints\)'
    )
    
    # Pattern to match warning messages about insufficient vectors
    warning_pattern_ivf = re.compile(
        r'\[IVF Path Warning\] Query\s+(\d+):\s+Insufficient tag-constrained vectors \((\d+)\s+<\s+(\d+)\)'
    )
    warning_pattern_graph = re.compile(
        r'\[Graph Path Warning\] Query\s+(\d+):\s+Insufficient tag-constrained vectors \((\d+)\s+<\s+(\d+)\)'
    )
    
    # Pattern to match query ID from debug headers
    query_id_pattern = re.compile(
        r'\[Debug (?:IVF|Graph) Path\] Query ID:\s+(\d+)'
    )
    
    total_results = {}  # query_id -> (actual, expected)
    warnings = []  # list of (query_id, path, actual, expected)
    query_paths = {}  # query_id -> path (IVF or Graph)
    
    with open(log_path, 'r') as f:
        current_query_id = None
        current_path = None
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Track current query ID and path
            match = query_id_pattern.search(line)
            if match:
                current_query_id = int(match.group(1))
                current_path = 'IVF' if 'IVF Path' in line else 'Graph'
                query_paths[current_query_id] = current_path
            
            # Check for warning messages
            match = warning_pattern_ivf.search(line)
            if match:
                query_id = int(match.group(1))
                actual = int(match.group(2))
                expected = int(match.group(3))
                warnings.append((query_id, 'IVF', actual, expected))
            
            match = warning_pattern_graph.search(line)
            if match:
                query_id = int(match.group(1))
                actual = int(match.group(2))
                expected = int(match.group(3))
                warnings.append((query_id, 'Graph', actual, expected))
            
            # Check for total results
            match = total_results_pattern.search(line)
            if match:
                actual = int(match.group(1))
                expected = int(match.group(2))
                if current_query_id is not None:
                    total_results[current_query_id] = (actual, expected)
    
    # Analysis results
    print("=" * 80)
    print("Log Analysis Report")
    print("=" * 80)
    print(f"Log file: {log_path}")
    print(f"Total queries with results: {len(total_results)}")
    print(f"Total warnings: {len(warnings)}")
    print()
    
    # Check for queries that didn't return k results
    insufficient_queries = []
    for query_id, (actual, expected) in sorted(total_results.items()):
        if actual < expected:
            insufficient_queries.append((query_id, actual, expected, query_paths.get(query_id, 'Unknown')))
    
    if insufficient_queries:
        print(f"❌ Found {len(insufficient_queries)} queries that did NOT return k results:")
        print("-" * 80)
        for query_id, actual, expected, path in insufficient_queries:
            print(f"  Query {query_id:5d} ({path:5s}): {actual} / {expected} results")
        print()
    else:
        print(f"✓ All {len(total_results)} queries returned the expected k results")
        print()
    
    # Show warnings
    if warnings:
        print(f"⚠ Found {len(warnings)} warning(s) about insufficient tag-constrained vectors:")
        print("-" * 80)
        for query_id, path, actual, expected in warnings:
            print(f"  Query {query_id:5d} ({path:5s}): {actual} < {expected} vectors available")
        print()
    
    # Summary statistics
    if total_results:
        all_expected = set(expected for _, (actual, expected) in total_results.items())
        print(f"Expected k values: {sorted(all_expected)}")
        
        actual_counts = defaultdict(int)
        for actual, expected in total_results.values():
            if actual == expected:
                actual_counts['full'] += 1
            else:
                actual_counts['partial'] += 1
        
        print(f"Queries with full results: {actual_counts['full']}")
        print(f"Queries with partial results: {actual_counts['partial']}")
        print()
    
    # Path distribution
    path_counts = defaultdict(int)
    for path in query_paths.values():
        path_counts[path] += 1
    
    print("Query path distribution:")
    for path, count in sorted(path_counts.items()):
        print(f"  {path}: {count} queries")
    
    print()
    print("=" * 80)
    
    return len(insufficient_queries) == 0

if __name__ == '__main__':
    log_path = '/home/mabojing/WorkForLab4/search_results/sift1m_hybrid_test/search_process.log'
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    all_ok = analyze_log(log_path)
    
    if not all_ok:
        sys.exit(1)

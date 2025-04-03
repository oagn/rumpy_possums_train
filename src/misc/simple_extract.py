#!/usr/bin/env python3
# Simple script to extract classification reports with classes 0, 12, and 99
import os
import re
import pandas as pd
import argparse

def extract_reports(log_file_path, output_dir):
    """Extract classification reports with classes 0, 12, and 99 from a log file."""
    print(f"Processing file: {log_file_path}")
    
    # Read the log file
    with open(log_file_path, 'r', errors='replace') as file:
        log_content = file.read()
    
    # Simple pattern for lines with class metrics
    class_0_pattern = r"\s*0\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    class_12_pattern = r"\s*12\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    class_99_pattern = r"\s*99\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    accuracy_pattern = r"\s*accuracy\s+(\d+\.\d+)\s+(\d+)"
    macro_avg_pattern = r"\s*macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    weighted_avg_pattern = r"\s*weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    
    # Find all matches
    class_0_matches = re.findall(class_0_pattern, log_content)
    class_12_matches = re.findall(class_12_pattern, log_content)
    class_99_matches = re.findall(class_99_pattern, log_content)
    accuracy_matches = re.findall(accuracy_pattern, log_content)
    macro_avg_matches = re.findall(macro_avg_pattern, log_content)
    weighted_avg_matches = re.findall(weighted_avg_pattern, log_content)
    
    # Get the minimum number of matches (they should all be the same)
    min_matches = min(
        len(class_0_matches), 
        len(class_12_matches), 
        len(class_99_matches),
        len(accuracy_matches),
        len(macro_avg_matches),
        len(weighted_avg_matches)
    )
    
    print(f"Found {min_matches} classification reports")
    
    # Create a list to hold all extracted metrics
    all_metrics = []
    
    # For each report, extract the metrics
    for i in range(min_matches):
        # Extract metrics for class 0
        all_metrics.append({
            'class': '0',
            'precision': float(class_0_matches[i][0]),
            'recall': float(class_0_matches[i][1]),
            'f1-score': float(class_0_matches[i][2]),
            'support': int(class_0_matches[i][3]),
            'report_index': i
        })
        
        # Extract metrics for class 12
        all_metrics.append({
            'class': '12',
            'precision': float(class_12_matches[i][0]),
            'recall': float(class_12_matches[i][1]),
            'f1-score': float(class_12_matches[i][2]),
            'support': int(class_12_matches[i][3]),
            'report_index': i
        })
        
        # Extract metrics for class 99
        all_metrics.append({
            'class': '99',
            'precision': float(class_99_matches[i][0]),
            'recall': float(class_99_matches[i][1]),
            'f1-score': float(class_99_matches[i][2]),
            'support': int(class_99_matches[i][3]),
            'report_index': i
        })
        
        # Extract accuracy metrics
        all_metrics.append({
            'class': 'accuracy',
            'precision': None,
            'recall': None,
            'f1-score': float(accuracy_matches[i][0]),
            'support': int(accuracy_matches[i][1]),
            'report_index': i
        })
        
        # Extract macro avg metrics
        all_metrics.append({
            'class': 'macro avg',
            'precision': float(macro_avg_matches[i][0]),
            'recall': float(macro_avg_matches[i][1]),
            'f1-score': float(macro_avg_matches[i][2]),
            'support': int(macro_avg_matches[i][3]),
            'report_index': i
        })
        
        # Extract weighted avg metrics
        all_metrics.append({
            'class': 'weighted avg',
            'precision': float(weighted_avg_matches[i][0]),
            'recall': float(weighted_avg_matches[i][1]),
            'f1-score': float(weighted_avg_matches[i][2]),
            'support': int(weighted_avg_matches[i][3]),
            'report_index': i
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'classification_reports.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(all_metrics)} metrics to {csv_path}")
    
    # Also create a summary with one row per report
    summaries = []
    for idx in range(min_matches):
        # Create a summary row for this report
        summary = {'report_index': idx}
        
        # Filter metrics for this report
        report_metrics = [m for m in all_metrics if m['report_index'] == idx]
        
        # Add metrics for each class
        for metric in report_metrics:
            class_name = metric['class']
            class_prefix = f"class_{class_name}_" if class_name in ['0', '12', '99'] else f"{class_name.replace(' ', '_')}_"
            
            # Add precision, recall, f1-score for this class
            if metric.get('precision') is not None:
                summary[class_prefix + 'precision'] = metric['precision']
            if metric.get('recall') is not None:
                summary[class_prefix + 'recall'] = metric['recall']
            summary[class_prefix + 'f1-score'] = metric['f1-score']
            summary[class_prefix + 'support'] = metric['support']
        
        summaries.append(summary)
    
    # Save summary to CSV
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(output_dir, 'classification_reports_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract classification reports from log files')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output_dir', type=str, default='reports', help='Directory to save the reports')
    
    args = parser.parse_args()
    
    extract_reports(args.log_file, args.output_dir)

if __name__ == "__main__":
    main() 
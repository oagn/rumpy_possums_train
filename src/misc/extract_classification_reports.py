#!/usr/bin/env python3
# Script to extract classification reports from log files and export them to CSV
import os
import re
import pandas as pd
import argparse
import glob
from pathlib import Path

# No need to update imports for this file since it doesn't import from other project modules

def parse_classification_report(report_text):
    """Parse a classification report text and convert it to a structured dictionary."""
    # Regular expressions to match different sections of the classification report
    lines = report_text.strip().split('\n')
    
    # Skip header lines
    data_lines = []
    for line in lines:
        if re.match(r'\s*\d+|accuracy|macro avg|weighted avg', line):
            data_lines.append(line)
    
    # Extract data for classes, accuracy, and averages
    metrics_data = []
    for line in data_lines:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 5:  # For class rows: [class, precision, recall, f1-score, support]
            if parts[0] == "accuracy":
                metrics_data.append({
                    'class': 'accuracy',
                    'precision': None,
                    'recall': None,
                    'f1-score': float(parts[1]),
                    'support': int(parts[2])
                })
            else:
                metrics_data.append({
                    'class': parts[0],
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3]),
                    'support': int(parts[4])
                })
    
    return metrics_data


def find_and_extract_reports(log_file_path):
    """Find all classification reports in a log file and extract them."""
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # Find all classification reports in the log file
    report_pattern = r"Classification report:\n(.*?)(?=\n\n|\Z)"
    reports = re.findall(report_pattern, log_content, re.DOTALL)
    
    # Parse each report
    parsed_reports = []
    for i, report in enumerate(reports):
        try:
            # Extract the stage/iteration information from nearby lines
            context_before = log_content.split(f"Classification report:\n{report}")[0]
            
            # Look for stage and iteration information
            iteration_match = re.search(r"Iteration (\d+)/\d+", context_before[-1000:])
            stage_match = re.search(r"Stage (\d+)/\d+", context_before[-1000:])
            
            iteration = iteration_match.group(1) if iteration_match else "unknown"
            stage = stage_match.group(1) if stage_match else "unknown"
            
            # If we couldn't find stage and iteration, look for other indicators
            if stage == "unknown" and iteration == "unknown":
                if "STAGE 1: Training initial wildlife model" in context_before[-1000:]:
                    stage = "wildlife"
                elif "STAGE 2: Fine-tuning on possum disease data" in context_before[-1000:]:
                    stage = "possum"
            
            # Parse the report
            parsed_report = parse_classification_report(report)
            
            # Add stage/iteration information to each row
            for row in parsed_report:
                row['iteration'] = iteration
                row['stage'] = stage
                row['report_index'] = i
            
            parsed_reports.extend(parsed_report)
        except Exception as e:
            print(f"Error parsing report {i}: {e}")
            continue
    
    return parsed_reports


def export_reports_to_csv(parsed_reports, output_dir, prefix="classification_reports"):
    """Export the parsed reports to CSV files."""
    if not parsed_reports:
        print("No reports found to export.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(parsed_reports)
    
    # Save all reports to a single CSV
    all_reports_path = os.path.join(output_dir, f"{prefix}_all.csv")
    df.to_csv(all_reports_path, index=False)
    print(f"Exported all classification reports to: {all_reports_path}")
    
    # Export separate CSVs for each iteration and stage
    # Group by iteration and stage
    for (iteration, stage), group_df in df.groupby(['iteration', 'stage']):
        # Skip groups with unknown iteration or stage if needed
        if iteration == "unknown" and stage == "unknown":
            continue
            
        # Create filename
        if iteration != "unknown":
            filename = f"{prefix}_iteration_{iteration}_stage_{stage}.csv"
        else:
            filename = f"{prefix}_stage_{stage}.csv"
        
        # Export CSV
        csv_path = os.path.join(output_dir, filename)
        group_df.to_csv(csv_path, index=False)
        print(f"Exported report for iteration {iteration}, stage {stage} to: {csv_path}")
    
    return all_reports_path


def main():
    parser = argparse.ArgumentParser(description='Extract classification reports from log files')
    parser.add_argument('--log_file', type=str, help='Path to the log file containing classification reports')
    parser.add_argument('--log_dir', type=str, help='Directory containing log files to process')
    parser.add_argument('--output_dir', type=str, default='classification_reports', 
                        help='Directory to save the extracted reports (default: classification_reports)')
    parser.add_argument('--prefix', type=str, default='classification_report',
                        help='Prefix for the CSV filenames (default: classification_report)')
    
    args = parser.parse_args()
    
    if not args.log_file and not args.log_dir:
        parser.error("Either --log_file or --log_dir must be specified")
    
    all_parsed_reports = []
    
    # Process single log file
    if args.log_file:
        if not os.path.exists(args.log_file):
            parser.error(f"Log file not found: {args.log_file}")
        print(f"Processing log file: {args.log_file}")
        parsed_reports = find_and_extract_reports(args.log_file)
        all_parsed_reports.extend(parsed_reports)
    
    # Process all log files in directory
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            parser.error(f"Log directory not found: {args.log_dir}")
        
        log_files = glob.glob(os.path.join(args.log_dir, "*.log")) + glob.glob(os.path.join(args.log_dir, "*.txt"))
        if not log_files:
            parser.error(f"No log files found in directory: {args.log_dir}")
        
        for log_file in log_files:
            print(f"Processing log file: {log_file}")
            parsed_reports = find_and_extract_reports(log_file)
            all_parsed_reports.extend(parsed_reports)
    
    # Export all parsed reports
    if all_parsed_reports:
        print(f"Found {len(all_parsed_reports)} report entries in total")
        export_reports_to_csv(all_parsed_reports, args.output_dir, args.prefix)
    else:
        print("No classification reports found in the provided log file(s)")


if __name__ == "__main__":
    main() 